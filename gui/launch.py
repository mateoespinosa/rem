import argparse
import os
import sys
from io import StringIO
import pandas as pd

from dnn_rem.rules.ruleset import Ruleset
from feature_explorer import FeatureExplorerComponent
from flexx import flx, ui
from rule_explorer import RuleExplorerComponent
from rule_list import RuleListComponent
from rule_statistics import RuleStatisticsComponent
from ruleset_loader import RulesetUploader
from uploader import FileUploader

from prediction_explorer import PredictComponent
from dnn_rem.experiment_runners.dataset_configs import (
    get_data_configuration, DatasetDescriptor, AVAILABLE_DATASETS
)
from pscript.stubs import window


################################################################################
## Helper Functions
################################################################################

def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Lunches CamRuleViz GUI for visualizing rulesets extracted using '
            'REM-D.'
        ),
    )

    parser.add_argument(
        '--rules',
        '-r',
        help=(
            "Valid .rules file containing serialized ruleset extracted by REM-D "
            "from some neural network and a given task."
        ),
        metavar="my_rules.rules",
        default=None,
    )

    parser.add_argument(
        '--data',
        help=(
            "Valid .csv file containing the dataset used to generate the "
            "given rules. If no descriptor is given, then it will assume all"
            "entires are reals and the last column is the target."
        ),
        metavar="data.csv",
        default=None,
    )

    parser.add_argument(
        '--data_descriptor',
        help=(
            "Valid name of supported dataset descriptor corresponding to the "
            "loaded data."
        ),
        metavar="descriptor_name",
        default=None,
    )

    parser.add_argument(
        '--show_tools',
        '-t',
        action="store_true",
        help=(
            "Whether or not we display Bokeh tools in summary plots."
        ),
        default=False,
    )

    parser.add_argument(
        '--max_entries',
        '-m',
        metavar="entries",
        type=int,
        help=(
            "Maximum number of entries to display in a summary histogram."
        ),
        default=15,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in the GUI.",
    )

    return parser


class FancyTabLayout(ui.TabLayout):
    pass


################################################################################
## Main Application
################################################################################


class RulesetLoadWindow(flx.PyComponent):
    def init(self):
        self.ruleset = Ruleset()
        self.dataset = None
        with ui.VBox(flex=1):
            ui.Widget(flex=1)  # Filler
            with ui.HBox(flex=1):
                ui.Widget(flex=1)  # Filler
                self.ruleset_label = flx.Label(
                    text="Load Ruleset",
                    css_class="ruleset-load-text",
                )
                self.upload_ruleset = RulesetUploader(
                    text="upload",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=1)  # Filler

            with ui.HBox(flex=1):
                ui.Widget(flex=1)  # Filler
                self.dataset_label = flx.Label(
                    text="Load Dataset",
                    css_class="ruleset-load-text",
                )
                self.upload_dataset = FileUploader(
                    text="upload",
                    binary=False,
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=1)  # Filler

            with ui.HBox(flex=1):
                ui.Widget(flex=1)  # Filler
                flx.Label(
                    text="Dataset Descriptor",
                    css_class="ruleset-load-text",
                )
                self.descriptor_selector = flx.ComboBox(
                    options=AVAILABLE_DATASETS + ["other"],
                    selected_index=len(AVAILABLE_DATASETS),
                )
                ui.Widget(flex=1)  # Filler

            with ui.HBox(flex=1):
                ui.Widget(flex=1)  # Filler
                self.continue_but = flx.Button(
                    text="Continue",
                    css_class='tool-bar-button',
                    disabled=True,
                )
                ui.Widget(flex=1)  # Filler

            ui.Widget(flex=1)  # Filler


    @flx.reaction('upload_ruleset.loading_error')
    def _ruleset_loading_error(self, *events):
        self.continue_but.set_disabled(True)
        self.ruleset_label.set_html(
            f'<span style="color: red;"> Load Ruleset </span>'
        )

    @flx.reaction('upload_ruleset.ruleset_load_ended')
    def _ruleset_loaded(self, *events):
        self.continue_but.set_disabled(False)
        self.ruleset_label.set_html(
            f'<span style="color: #4CAF50;"> Load Ruleset </span>'
        )
        self.ruleset = events[-1]['ruleset']

    @flx.reaction('upload_dataset.reading_error')
    def _dataset_loading_error(self, *events):
        self.dataset_label.set_html(
            f'<span style="color: red;"> Load Dataset </span>'
        )

    @flx.reaction('upload_dataset.file_loaded')
    def _dataset_loaded(self, *events):
        self.continue_but.set_disabled(False)
        data_str = events[-1]['filedata']
        try:
            df = pd.read_csv(StringIO(data_str), sep=",")
            descriptor_ind = self.descriptor_selector.selected_index
            if descriptor_ind < len(AVAILABLE_DATASETS):
                self.dataset = get_data_configuration(
                    AVAILABLE_DATASETS[descriptor_ind]
                )
            else:
                self.dataset = DatasetDescriptor()
            self.dataset.process_dataframe(df)

            self.dataset_label.set_html(
                f'<span style="color: #4CAF50;"> Load Dataset </span>'
            )
        except Exception as e:
            self.dataset_label.set_html(
                f'<span style="color: red;"> Load Dataset </span>'
            )
            self.dataset = None
            self.data_loading_error(e)

    @flx.emitter
    def data_loading_error(self, e):
        return {'error': e}

    @flx.emitter
    def ruleset_loading_error(self, e):
        return {'error': e}

    @flx.reaction('continue_but.pointer_click')
    def _continue_click(self, *events):
        for event in events:
            self.loading_complete()

    @flx.emitter
    def loading_complete(self):
        return {
            'ruleset': self.ruleset,
            'dataset': self.dataset,
        }


class RuleVizWindow(flx.PyComponent):
    tabs = flx.ListProp(settable=True)

    def init(self):
        with flx.VBox(title="RuleViz"):
            with flx.HBox():
                flx.Label(
                    text="RuleViz",
                    style=(
                        "font-family: 'Josefin Sans', sans-serif;"
                        "font-size: 500%;"
                        "padding-top: 20px;"
                    ),
                )
            with FancyTabLayout(flex=1):
                self.add_tab(RuleStatisticsComponent())
                self.add_tab(PredictComponent(
                    self.root.state.ruleset,
                ))
                self.add_tab(RuleExplorerComponent())
                if self.root.state.dataset is not None:
                    # Then we add the feature explorer window
                    self.add_tab(FeatureExplorerComponent())
                self.add_tab(RuleListComponent(
                    self.root.state.ruleset,
                ))

    @flx.action
    def add_tab(self, widget):
        self._mutate_tabs(
            [widget],
            'insert',
            len(self.tabs),
        )

    @flx.reaction("tabs*.ruleset_update")
    def update_view(self, *events):
        print(
            "Time to update other tabs given that the ruleset got updated:",
            events
        )
        for event in events:
            for i, tab in enumerate(self.tabs):
                if tab.id == event["source_id"]:
                    # Then nothing to do here
                    continue

                # As this tab to get updated!
                tab.perform_update(event)


class CamRuleState(object):
    def __init__(
        self,
        ruleset=None,
        dataset=None,
        show_tools=False,
        max_entries=15,
        merge_branches=False,
    ):
        self.ruleset = ruleset
        if ruleset is not None:
            self.original_ruleset = ruleset.copy()
        else:
            self.original_ruleset = None
        self.dataset = dataset
        self.show_tools = show_tools
        self.max_entries = max_entries
        self._feature_ranges = {}
        self.merge_branches = merge_branches

    def set_ruleset(self, ruleset):
        self.ruleset = ruleset
        if ruleset is not None:
            self.original_ruleset = ruleset.copy()

    def reset_ruleset(self):
        if self.original_ruleset is not None:
            self.ruleset = self.original_ruleset.copy()

    def merge_ruleset(self, other):
        if self.ruleset is not None:
            self.ruleset.merge(other)
        else:
            self.set_ruleset(other)

    def get_feature_range(self, feature, empirical=True):
        if empirical and (feature in self._feature_ranges):
            return self._feature_ranges[feature]
        if self.dataset is None:
            # Then we cannot bound this guy
            return (-float("inf"), float("inf"))
        (min_val, max_val) = self.dataset.get_feature_ranges(feature)
        if empirical and (min_val in [float("inf"), -float("inf")]):
            # Then we will use the empirical limit for visualization
            # purposes
            min_val = min(self.dataset.data[feature])
        if empirical and (max_val in [float("inf"), -float("inf")]):
            # Then we will use the empirical limit for visualization
            # purposes
            max_val = max(self.dataset.data[feature])
        result = (min_val, max_val)
        if empirical:
            self._feature_ranges[feature] = result
        return result


class CamRuleViz(flx.PyComponent):
    windows = flx.ListProp(settable=True)
    current_window = flx.IntProp(0, settable=True)

    def init(
        self,
        ruleset=None,
        dataset=None,
        show_tools=False,
        max_entries=15,
        merge_branches=False,
    ):
        self.state = CamRuleState(
            ruleset=ruleset,
            dataset=dataset,
            show_tools=show_tools,
            max_entries=max_entries,
            merge_branches=merge_branches,
        )
        with ui.VBox(title="RuleViz"):
            with ui.StackLayout(flex=1) as self.stack:
                self._mutate_windows(
                    [
                        RulesetLoadWindow() if ruleset is None
                        else RuleVizWindow()
                    ],
                    'insert',
                    len(self.windows),
                )

    @flx.reaction('!windows*.loading_complete')
    def complete_loading(self, event):
        self.state.set_ruleset(event['ruleset'])
        self.state.dataet = event.get('dataset', None)

        # Now build up the rest of the application given that we have provided
        # consent
        with self:
            with self.stack:
                self._mutate_windows(
                    [RuleVizWindow()],
                    'insert',
                    len(self.windows),
                )
        self.set_current_window(
            (self.current_window + 1) % len(self.windows)
        )
        self.stack.set_current(self.current_window)


def main():
    flx.assets.associate_asset(__name__, 'https://d3js.org/d3.v3.min.js')
    parser = build_parser()
    args = parser.parse_args()
    if args.rules is not None:
        ruleset = Ruleset()
        ruleset.from_file(args.rules)
    else:
        ruleset = None

    if args.data is not None:
        if args.data_descriptor is not None:
            dataset = get_data_configuration(args.data_descriptor)
        else:
            dataset = DatasetDescriptor()
        dataset.read_data(args.data)
    else:
        dataset = None

    with open(os.path.join(os.path.dirname(__file__), 'style.css')) as f:
        style = f.read()
    flx.assets.associate_asset(__name__, 'style.css', style)

    app = flx.App(
        CamRuleViz,
        ruleset,
        dataset,
        args.show_tools,
        args.max_entries,
    )

    app.launch('browser')  # show it now in a browser
    flx.run()  # enter the mainloop

    return 0

################################################################################
## Program Entry Point
################################################################################

if __name__ == '__main__':
    sys.exit(main())

