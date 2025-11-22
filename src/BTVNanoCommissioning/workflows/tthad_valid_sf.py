import collections, gc
import os
import uproot
import numpy as np, awkward as ak
import hist
import re

from coffea import processor
from coffea.analysis_tools import Weights

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)
from BTVNanoCommissioning.helpers.func import update, dump_lumi, PFCand_link
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writter,
)
from BTVNanoCommissioning.utils.selection import HLT_helper, jet_id, btag_mu_idiso, MET_filters, mu_idiso, ele_cuttightid

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="tt_had",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ### Added selection for ttbar hadronic
        self.ttaddsel = selectionModifier
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events)
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=["jet0", "jet1","jet2","jet3","jet4","jet5"],
                hist_collections=["common", "fourvec", "tthad"],
            )

        if isRealData:
            output["sumw"] = len(events)
        else:
            output["sumw"] = ak.sum(events.genWeight)
        
        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)

        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        # Run-dependent PFHT trigger paths, taken from https://twiki.cern.ch/twiki/bin/view/CMS/TopTrigger
        PFHT_TRIGGER_MAP = {
            # -------------------------
            # Run 2
            # -------------------------
            "2016": {
                # Eras: B, C, D, E, F, G, H
                "default": [
                    "PFHT400_SixJet30_DoubleBTagCSV_p056",
                    "PFHT450_SixJet40_BTagCSV_p056",
                ],
            },
            "2017": {
                # Era B
                "B": [
                    "PFHT380_SixJet32_DoubleBTagCSV_p075",
                    "PFHT430_SixJet40_BTagCSV_p080",
                ],
                # Eras C, D, E, F
                "default": [
                    "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2",
                    "PFHT430_SixPFJet40_PFBTagCSV_1p5",
                ],
            },
            "2018": {
                # Eras A, B, C, D
                "default": [
                    "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
                    "PFHT430_SixPFJet40_PFBTagDeepCSV_1p5",
                ],
            },
            # -------------------------
            # Run 3
            # -------------------------
            "2022": {
                # Eras C, D, E, F, G
                "default": [
                    "PFHT400_SixPFJet32_DoublePFBTagDeepJet_2p94",
                    "PFHT450_SixPFJet36_PFBTagDeepJet_1p59",
                ],
            },
            "2023": {
                # Eras C, D  
                "default": [
                    "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                    "PFHT450_SixPFJet36_PFBTagDeepJet_1p59",

                ],
            },
        }

        def pfht_triggers(year: str, dataset: str):
            """
            Return the PFHT trigger paths for a given data-taking year and dataset name.
            """
            run_letter = None
        
            # Primary pattern: Run2017B, Run2018A, etc.
            match = re.search(r"Run20\d{2}([A-Z])", dataset)
            if match:
                run_letter = match.group(1)
            else:
                # Fallback if the dataset tokenization is weird
                for token in dataset.replace("_", "").split("/"):
                    if token.startswith("Run20") and len(token) >= 7:
                        run_letter = token[-1]
                        break
        
            # ---- Retrieve trigger mapping ----
            year_map = PFHT_TRIGGER_MAP.get(str(year), {})
        
            # If it recognizes a valid run letter and it's in the map, uses it
            if run_letter and run_letter in year_map:
                return year_map[run_letter]
        
            # Otherwise, use the year's default
            if "default" in year_map:
                return year_map["default"]
        
            # Final fallback if there is no match
            return [
                "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
                "PFHT450_SixPFJet36_PFBTagDeepJet_1p59",
            ]

        
        triggers = pfht_triggers(self._year, dataset)
        req_trig = HLT_helper(events, triggers)

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        events.Muon = events.Muon[
            (events.Muon.pt > 20) & mu_idiso(events, self._campaign)
        ]
        req_muon = ak.count(events.Muon.pt, axis=1) == 0

        ## Electron cuts
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        events.Electron = events.Electron[
            (events.Electron.pt > 20) & ele_cuttightid(events, self._campaign)
        ]
        req_ele = ak.count(events.Electron.pt, axis=1) == 0

        # Jet cuts
        # Using JetID
        jet_mask = jet_id(events, self._campaign)
        
        # Handle DeltaR with leptons
        has_mu = ak.num(events.Muon) > 0
        has_ele = ak.num(events.Electron) > 0
        
        dr_mu  = ~has_mu | ak.all(events.Jet.metric_table(events.Muon) > 0.4, axis=2)
        dr_ele = ~has_ele | ak.all(events.Jet.metric_table(events.Electron) > 0.4, axis=2)
        
        jetsel = jet_mask & dr_mu & dr_ele
        jetsel = ak.fill_none(jetsel, False)
        
        # Jet requirement
        event_jet = events.Jet[jetsel]
        req_jets = ak.num(event_jet.pt) >= 6

        # B-jet requirement
        bjet = event_jet[event_jet.btagDeepFlavB > 0.058] #T=0.7183
        req_bjets = ak.num(bjet.pt) >= 2 

        event_weights = (
            ak.to_numpy(events.genWeight) if not isRealData else np.ones(len(events))
        )

        event_level = (
            req_trig &
            req_lumi &
            req_muon &
            req_ele &
            req_jets &
            req_bjets
        )
        event_level = ak.fill_none(event_level, False)

        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        pruned_ev = events[event_level]
        
        # Store selected muons and electrons
        pruned_ev["SelMuon"] = events.Muon[event_level]
        pruned_ev["SelElectron"] = events.Electron[event_level]
        
        # Store selected jets and limit to first 6
        pruned_ev["SelJet"] = event_jet[event_level][:, :6]
        pruned_ev["njet"] = ak.count(pruned_ev.SelJet.pt, axis=1)
        
        # Store selected b-jets
        pruned_ev["SelBJet"] = bjet[event_level]
        pruned_ev["nbjet"] = ak.count(pruned_ev.SelBJet.pt, axis=1)

        # If PFCands exists, create a dict of PFCands per jet index
        if "PFCands" in events.fields:
            pruned_ev["JetPFCandsAll"] = {}
            pruned_ev["PFCandsPerJet"] = {}
        
            for i in range(6):
                # jet indices for i-th jet
                pruned_ev["JetPFCandsAll"][i] = jetindx[:, i]
        
                # PFCands linked to i-th jet
                pruned_ev["PFCandsPerJet"][i] = events[event_level].PFCands[
                    events[event_level]
                    .JetPFCands[
                        events[event_level].JetPFCands.jetIdx
                        == pruned_ev["JetPFCandsAll"][i][event_level]
                    ]
                    .pFCandsIdx
                ]

        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)

        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )

        # Output arrays
        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )
        
        print("\n \n Returning output:", output.keys())

        return {dataset: output}
        
    def postprocess(self, accumulator):
        return accumulator
