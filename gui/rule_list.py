from flexx import flx, ui
from dnn_rem.rules.rule import Rule
from gui_window import CamvizWindow


def _clause_to_str(clause):
    result = ""
    terms = []
    for term in clause.terms:
        op = '&leq;' if term.operator == '<=' else term.operator
        terms.append(
            f"(<span style='font-weight: bold; color: #ff6666;'>"
            f"{term.variable}"
            f"</span> {op} {term.threshold})"
        )
    return " <span style='color: #758a7d;'>AND</span> ".join(terms)


class RuleView(flx.Widget):
    CSS = """
    .flx-RuleView {
        border-style: dashed;
        border-color: black;
        border-width: thin;
    }
    .flx-RuleView:hover {
        background-color: #eefafe;
    }"""

    precedent = flx.StringProp(settable=True)
    conclusion = flx.StringProp(settable=True)
    score = flx.FloatProp(settable=True)
    idx = flx.IntProp(settable=True)

    def init(self):
        with flx.HBox():
            self.index_label = flx.Label(
                text=lambda: f'{self.idx + 1}. ',
                flex=0,
                style=(
                    "font-weight: bold;"
                    'background-color: #d4cdc7;'
                ),
            )
            self.label = flx.Label(
                flex=(1, 0),
                style=(
                    'overflow-x: scroll;'
                ),
            )
            self.label.set_html(
                "<span class='camviz_tooltip'>"
                "<b style=\"font-family: 'Source Code Pro', monospace;\"> IF</b>"
                f'<span style="font-family:\'Inconsolata\';">'
                f' {self.precedent}</span> '
                f"<b style=\"font-family: 'Source Code Pro', monospace;\">THEN"
                f'</b> <i>{self.conclusion}</i>'
                f'<span class="camviz_tooltiptext">Score: {self.score}</span>'
                "</span>"
            )
            self.delete_button = flx.Button(text="delete", flex=0)

    @flx.emitter
    def rule_removed(self):
        return {
            'conclusion': self.conclusion,
            'idx': self.idx,
            'precedent': self.precedent,
            'score:': self.score,
        }

    @flx.reaction('delete_button.pointer_click')
    def delete_rule(self, *events):
        self.rule_removed()
        self.set_parent(None)


class ClassRuleList(flx.PyComponent):
    rules = flx.ListProp(settable=True)
    class_id = flx.IntProp(settable=True)

    def init(self, rule):
        self.rule_obj = rule
        self.clause_ordering = []
        with ui.GroupWidget(
            title=(
                f'Class: {self.rule_obj.conclusion} '
                f'({len(self.rule_obj.premise)} '
                f'rule{"" if len(self.rule_obj.premise) == 1 else "s"})'
            ),
            style=(
                'overflow-y: scroll;'
            )
        ) as self.class_group:
            for idx, clause in enumerate(sorted(
                self.rule_obj.premise,
                key=lambda x: x.score,
            )):
                self.clause_ordering.append(clause)
                self.add_rule(
                    idx,
                    clause,
                    self.rule_obj.conclusion,
                    clause.score,
                )

    @flx.action
    def _add_rule(self, rule):
        self._mutate_rules([rule], 'insert', len(self.rules))

    @flx.action
    def add_rule(self, idx, clause, conclusion, score):
        new_rule = RuleView(
            idx=idx,
            precedent=_clause_to_str(clause),
            conclusion=conclusion,
            score=score,
        )
        self._add_rule(new_rule)

    @flx.action
    def _remove_rule(self, idx):
        self._mutate_rules(1, 'remove', idx)

    @flx.emitter
    def ruleset_update(self):
        return {
            'class_id': self.class_id,
        }

    @flx.reaction('rules*.rule_removed')
    def remove_rule(self, *events):
        # Time to remove the rule from our ruleset
        for event in events:
            clause_idx = event["idx"]
            self.root.state.ruleset.remove_rule(
                Rule(
                    premise=set([self.clause_ordering[clause_idx]]),
                    conclusion=self.rule_obj.conclusion,
                )
            )

            # Remove it from our ordering of the different rules
            self.clause_ordering.pop(clause_idx)

            # And update the IDs of all entries that came after this one
            for i, rule in enumerate(self.rules):
                if i <= clause_idx:
                    continue
                rule.set_idx(rule.idx - 1)

            # Remove the rule entry from our list of rules
            self._remove_rule(clause_idx)

        # Update the title
        self.class_group.set_title(
            f'Class: {self.rule_obj.conclusion} '
            f'({len(self.rule_obj.premise)} '
            f'rule{"" if len(self.rule_obj.premise) == 1 else "s"})'
        )

        # Finally, emit an even that will tell all other windows to update
        # as needed
        self.ruleset_update()

    @flx.action
    def reset(self):
        self.class_group.set_title(
            f'Class: {self.rule_obj.conclusion} '
            f'({len(self.rule_obj.premise)} '
            f'rule{"" if len(self.rule_obj.premise) == 1 else "s"})'
        )
        self._mutate_rules([])
        self.clause_ordering = []
        with self:
            with self.class_group:
                for idx, clause in enumerate(sorted(
                    self.rule_obj.premise,
                    key=lambda x: x.score,
                )):
                    self.clause_ordering.append(clause)
                    self.add_rule(
                        idx,
                        clause,
                        self.rule_obj.conclusion,
                        clause.score,
                    )


class RuleListComponent(CamvizWindow):
    class_rulesets = flx.ListProp(settable=True)
    class_buttons = flx.ListProp(settable=True)

    def init(self):
        ruleset = self.root.state.ruleset
        classes = sorted(ruleset.rules, key=lambda x: x.conclusion)
        with ui.HBox(title="Rule Editor") as tab:
            with ui.VBox(
                style=(
                    'overflow-y: scroll;'
                    'overflow-x: scroll;'
                )
            ) as self.box_pannel:
                self.pannel_title = flx.Label(
                    text="Classes",
                    style=(
                        'font-weight: bold;'
                        'font-size: 175%;'
                    )
                )
                for class_idx, rule in enumerate(classes):
                    new_button = ui.Button(
                        text=rule.conclusion,
                        style=(
                            'font-weight: bold;'
                            'font-size: 150%;'
                        )
                    )
                    self._mutate_class_buttons(
                        [new_button],
                        'insert',
                        len(self.class_buttons),
                    )

                # And add an empty widget as a space filler
                ui.Widget(flex=1)

            with ui.StackLayout(
                flex=1,
                style=(
                    'overflow-y: scroll;'
                    'overflow-x: scroll;'
                )
            ) as self.stack:
                for i, rule in enumerate(classes):
                    new_set = ClassRuleList(rule)
                    self.class_buttons[i].window_idx = i
                    self._mutate_class_rulesets(
                        [new_set],
                        'insert',
                        len(self.class_rulesets),
                    )

    @flx.reaction('class_buttons*.pointer_down')
    def _stacked_current(self, *events):
        button = events[-1].source
        self.stack.set_current(button.window_idx)


    @flx.reaction('class_rulesets*.ruleset_update')
    def bypass_update(self, *events):
        for event in events:
            event = event.copy()
            event["source_id"] = self.id
            self.ruleset_update(event)


    @flx.action
    def reset(self):
        for class_ruleset in self.class_rulesets:
            class_ruleset.reset(event)
