from flexx import flx, ui
from dnn_rem.rules.term import TermOperator
from gui_window import CamvizWindow


def _get_rules_stats(ruleset):
    stats = {}
    stats["Total number of rules"] = sum([
        len(rule.premise) for rule in ruleset.rules
    ])
    stats["Number of output classes"] = len(ruleset.output_class_map)
    stats["Maximum number of rules for a single class"] = max([
        len(rule.premise) for rule in ruleset.rules
    ])
    stats["Minimum number of rules for a single class"] = min([
        len(rule.premise) for rule in ruleset.rules
    ])
    stats["Number of clauses in longest rule"] = max([
        len(clause.terms) for rule in ruleset.rules
        for clause in rule.premise
    ])
    stats["Number of clauses in shortest rule"] = min([
        len(clause.terms) for rule in ruleset.rules
        for clause in rule.premise
    ])
    feature_freq = {}
    term_freq = {}
    for rule in ruleset.rules:
        output = rule.conclusion
        for clause in rule.premise:
            for term in clause.terms:
                var_name = term.variable
                op = (
                    '&leq;' if term.operator == TermOperator.LessThanEq
                    else str(term.operator)
                )
                full_term = (var_name, op, term.threshold)
                feature_freq[var_name] = feature_freq.get(var_name, 0) + 1
                term_freq[full_term] = term_freq.get(full_term, 0) + 1
    stats["Total number of input features"] = len(ruleset.feature_names)
    stats["Total number of input features used in rules"] = len(
        feature_freq.keys()
    )
    sorted_freq_features = sorted(
        list(feature_freq.items()),
        key=lambda x: -feature_freq[x[0]]
    )
    stats["Top 5 most used features"] = "<ol>" + (
        "\n".join(map(
            lambda x: f'<li> <b>{x[0]}:</b> {x[1]}</li>',
            sorted_freq_features[:5]
        ))
    ) + "</ol>"
    stats["Bottom 5 least used features"] = "<ol>" + (
        "\n".join(map(
            lambda x: f'<li> <b>{x[0]}:</b> {x[1]}</li>',
            sorted_freq_features[-5:]
        ))
    ) + "</ol>"
    stats["Total number of unique terms"] = len(term_freq)
    sorted_term_freq = sorted(
        list(term_freq.items()),
        key=lambda x: -term_freq[x[0]]
    )
    stats["Top 5 most used terms"] = "<ol>" + (
        "\n".join(map(
            lambda x: (
                f'<li> <b>{x[0][0]}</b> {x[0][1]} {x[0][2]} ({x[1]} '
                f'time{"" if x[1] == 1 else "s"})</li>'
            ),
            sorted_term_freq[:5]
        ))
    ) + "</ol>"
    return stats


class RuleSummaryComponent(CamvizWindow):
    lines = flx.ListProp(settable=True)

    def init(self):
        with flx.Widget(
            title="Summary",
            style='overflow-y: scroll;',
        ):
            ruleset_stats = _get_rules_stats(self.root.state.ruleset)
            for stat_name, stat_val in ruleset_stats.items():
                self.add_line(f'<b>{stat_name}:</b> {stat_val}')

    @flx.action
    def add_line(self, text):
        new_label = flx.Label(
            style=(
                'overflow-x: scroll;'
            )
        )
        new_label.set_html(text)
        self._mutate_lines(
            [new_label],
            'insert',
            len(self.lines),
        )

    @flx.action
    def reset(self):
        ruleset_stats = _get_rules_stats(self.root.state.ruleset)
        for i, (stat_name, stat_val) in enumerate(ruleset_stats.items()):
            self.lines[i].set_html(f'<b>{stat_name}:</b> {stat_val}')
