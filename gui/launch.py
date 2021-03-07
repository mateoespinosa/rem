import argparse
import os
import sys

from dnn_rem.rules.ruleset import Ruleset
from feature_explorer import FeatureExplorerComponent
from flexx import flx, ui
from rule_explorer import RuleExplorerComponent
from rule_list import RuleListComponent
from rule_statistics import RuleStatisticsComponent
from prediction_explorer import PredictComponent
import pandas as pd
from dnn_rem.experiment_runners.dataset_configs import (
    get_data_configuration, DatasetDescriptor
)


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
        'rules',
        help=(
            "Valid .rules file containing serialized ruleset extracted by REM-D "
            "from some neural network and a given task."
        ),
        metavar="my_rules.rules",
    )

    parser.add_argument(
        'data_path',
        help=(
            "Valid .csv file containing the dataset used to generate the "
            "given rules. If no descriptor is given, then it will assume all"
            "entires are reals and the last column is the target."
        ),
        metavar="data.csv",
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
    """ Fancy version of the TabLayout with nicer looking aesthetics.
    """
    pass

################################################################################
## Main Application
################################################################################


class CamRuleState(object):
    def __init__(self, ruleset, dataset, show_tools=False, max_entries=15):
        self.ruleset = ruleset
        self.dataset = dataset
        self.show_tools = show_tools
        self.max_entries = max_entries
        self._feature_ranges = {}

    def get_feature_range(self, feature):
        if feature in self._feature_ranges:
            return self._feature_ranges[feature]
        (min_val, max_val) = self.dataset.get_feature_ranges(feature)
        if min_val in [float("inf"), -float("inf")]:
            # Then we will use the empirical limit for visualization
            # purposes
            min_val = min(self.dataset.data[feature])
        if max_val in [float("inf"), -float("inf")]:
            # Then we will use the empirical limit for visualization
            # purposes
            max_val = max(self.dataset.data[feature])
        result = (min_val, max_val)
        self._feature_ranges[feature] = result
        return result


class CamRuleViz(flx.PyComponent):
    windows = flx.ListProp(settable=True)

    def init(self, ruleset, dataset, show_tools, max_entries):
        self.state = CamRuleState(
            ruleset=ruleset,
            dataset=dataset,
            show_tools=show_tools,
            max_entries=max_entries,
        )
        with flx.VBox(title="CamRuleViz"):
            with flx.HBox():
                flx.Label(
                    text="RuleViz",
                    style=(
                        "font-family: 'Josefin Sans', sans-serif;"
                        "font-size: 500%;"
                        "padding-top: 20px;"
                    ),
                )
            with FancyTabLayout(flex=1) as self.tabs:
                self.add_window(RuleStatisticsComponent())
                self.add_window(PredictComponent(
                    self.state.ruleset,
                ))
                self.add_window(RuleExplorerComponent())
                self.add_window(FeatureExplorerComponent())
                self.add_window(RuleListComponent(
                    self.state.ruleset,
                ))

    @flx.action
    def add_window(self, widget):
        self._mutate_windows(
            [widget],
            'insert',
            len(self.windows),
        )

    @flx.reaction("windows*.ruleset_update")
    def update_view(self, *events):
        print(
            "Time to update other windows given that the ruleset got updated:",
            events
        )
        for event in events:
            for i, window in enumerate(self.windows):
                if window.id == event["source_id"]:
                    # Then nothing to do here
                    continue

                # As this window to get updated!
                window.perform_update(event)


def main():
    flx.assets.associate_asset(__name__, 'https://d3js.org/d3.v3.min.js')
    parser = build_parser()
    args = parser.parse_args()
    ruleset = Ruleset()
    ruleset.from_file(args.rules)
    if args.data_descriptor is not None:
        dataset = get_data_configuration(args.data_descriptor)
    else:
        dataset = DatasetDescriptor()
    dataset.read_data(args.data_path)

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

