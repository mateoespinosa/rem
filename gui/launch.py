import argparse
import sys
import os

from flexx import flx, ui
from dnn_rem.rules.ruleset import Ruleset
from rule_list import RuleListComponent
from rule_summary import RuleSummaryComponent


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
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in the GUI.",
    )

    return parser


################################################################################
## Main Application
################################################################################


class CamRuleState(object):
    def __init__(self, ruleset):
        self.ruleset = ruleset


class CamRuleViz(flx.PyComponent):
    windows = flx.ListProp(settable=True)

    def init(self, ruleset):
        self.state = CamRuleState(ruleset=ruleset)
        with flx.VBox(title="CamRuleViz"):
            with flx.HBox():
                flx.Label(
                    text="RuleViz",
                    style=(
                        "font-family: 'Josefin Sans', sans-serif;"
                        "font-size: 400%;"
                    ),
                )
            with ui.TabLayout(flex=1) as self.tabs:
                self.add_window(RuleSummaryComponent())
                self.add_window(RuleListComponent())

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

    with open(os.path.join(os.path.dirname(__file__), 'style.css')) as f:
        style = f.read()
    flx.assets.associate_asset(__name__, 'style.css', style)

    app = flx.App(
        CamRuleViz,
        ruleset,
    )

    app.launch('browser')  # show it now in a browser
    flx.run()  # enter the mainloop

    return 0

################################################################################
## Program Entry Point
################################################################################

if __name__ == '__main__':
    sys.exit(main())

