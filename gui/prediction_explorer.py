from flexx import flx, ui
from gui_window import CamvizWindow
from collections import defaultdict
from pscript.stubs import d3, window
from dnn_rem.rules.term import TermOperator
from dnn_rem.rules.ruleset import Ruleset
from sklearn.neighbors import KernelDensity
import numpy as np
from rule_statistics import _CLASS_PALETTE
from rule_explorer import HierarchicalTreeViz, ruleset_hierarchy_tree
from rule_list import ClassRuleList
import pprint

###############################################################################
## Graphical Path Visualization Component
###############################################################################

def _get_activated_ruleset(ruleset, activations):
    vector = np.zeros([len(ruleset.feature_names)])
    for i, feature_name in enumerate(ruleset.feature_names):
        vector[i] = activations.get(feature_name, 0)
    [prediction], [activated_rules] = ruleset.predict_and_explain(
        X=vector,
        use_label_names=True,
    )
    return prediction, Ruleset(
        rules=activated_rules,
        feature_names=ruleset.feature_names,
        output_class_names=ruleset.output_class_names(),
    )


def _filter_ruleset(ruleset, cls_name):
    rules = []
    for rule in ruleset.rules:
        if rule.conclusion == cls_name:
            rules.append(rule)
    return Ruleset(
        rules=rules,
        feature_names=ruleset.feature_names,
        output_class_names=ruleset.output_class_names(),
    )


def _get_prediction_confidence(ruleset, cls_name):
    result = 0
    tot_sum = 0
    for rule in ruleset.rules:
        if rule.conclusion == cls_name:
            for clause in rule.premise:
                tot_sum += clause.score
                result += clause.score * clause.confidence
    return result / tot_sum if tot_sum else 0


class PredictionPathComponent(flx.PyWidget):
    predicted_val = flx.StringProp("", settable=True)

    def init(self, ruleset):
        self.ruleset = ruleset
        with ui.VSplit(flex=1, css_class='prediction-path-container'):
            self.tree_view = HierarchicalTreeViz(
                data=ruleset_hierarchy_tree(self.ruleset),
                fixed_node_radius=5,
                class_names=self.ruleset.output_class_names(),
                flex=0.75,
                branch_separator=1,
            )
            with ui.HBox(
                css_class="prediction-result-tool-bar",
                flex=0.07,
            ):
                ui.Widget(flex=1)  # Filler
                self.collapse_button = flx.Button(
                    text="Collapse Tree",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.expand_button = flx.Button(
                    text="Expand Tree",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.fit_button = flx.Button(
                    text="Fit to Screen",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.only_positive = flx.CheckBox(
                    text="Only Predicted Class",
                    css_class='tool-bar-checkbox',
                    checked=True,
                )
                ui.Widget(flex=1)  # Filler
            ui.Widget(flex=0.15)  # Filler

    @flx.action
    def set_ruleset(self, ruleset):
        self.ruleset = ruleset
        self._update()

    def _update(self, checked=None):
        if checked is None:
            checked = self.only_positive.checked
        if checked:
            ruleset = _filter_ruleset(
                ruleset=self.ruleset,
                cls_name=self.predicted_val,
            )
        else:
            ruleset = self.ruleset
        self.tree_view.set_data(
            ruleset_hierarchy_tree(ruleset)
        )

    @flx.reaction('expand_button.pointer_click')
    def _expand_tree(self, *events):
        self.tree_view.expand_tree()

    @flx.reaction('fit_button.pointer_click')
    def _fit_tree(self, *events):
        self.tree_view.zoom_fit()

    @flx.reaction('collapse_button.pointer_click')
    def _collapse_tree(self, *events):
        self.tree_view.collapse_tree()

    @flx.reaction('only_positive.user_checked')
    def _check_positive(self, *events):
        self._update(checked=events[-1]["new_value"])


###############################################################################
## Rule List Visualization Component
###############################################################################

class NumberEdit(flx.Widget):
    DEFAULT_MIN_SIZE = 100, 28

    CSS = """
    .flx-LineEdit {
        color: #333;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        border: 1px solid #aaa;
        margin: 2px;
    }
    .flx-LineEdit:focus  {
        outline: none;
        box-shadow: 0px 0px 3px 1px rgba(0, 100, 200, 0.7);
    }
    """

    ## Properties

    num = flx.IntProp(settable=True, doc="""
        The current num of the line edit. Settable. If this is an empty
        string, the placeholder_num is displayed instead.
        """)

    password_mode = flx.BoolProp(False, settable=True, doc="""
        Whether the insered num should be hidden.
        """)

    placeholder_num = flx.StringProp(settable=True, doc="""
        The placeholder num (shown when the num is an empty string).
        """)

    autocomp = flx.TupleProp(settable=True, doc="""
        A tuple/list of strings for autocompletion. Might not work in all browsers.
        """)

    disabled = flx.BoolProp(False, settable=True, doc="""
        Whether the line edit is disabled.
        """)

    ## Methods, actions, emitters

    def _create_dom(self):
        global window

        # Create node element
        node = window.document.createElement('input')
        node.setAttribute('type', 'number')
        node.type = 'number'
        node.setAttribute('list', self.id)

        self._autocomp = window.document.createElement('datalist')
        self._autocomp.id = self.id
        node.appendChild(self._autocomp)

        f1 = lambda: self.user_num(self.node.value)
        self._addEventListener(node, 'input', f1, False)
        self._addEventListener(node, 'blur', self.user_done, False)
        #if IE10:
        #    self._addEventListener(self.node, 'change', f1, False)
        return node

    @flx.emitter
    def user_num(self, num):
        """ Event emitted when the user edits the num. Has ``old_value``
        and ``new_value`` attributes.
        """
        d = {'old_value': self.num, 'new_value': num}
        self.set_num(num)
        return d

    @flx.emitter
    def user_done(self):
        """ Event emitted when the user is done editing the num, either by
        moving the focus elsewhere, or by hitting enter.
        Has ``old_value`` and ``new_value`` attributes (which are the same).
        """
        d = {'old_value': self.num, 'new_value': self.num}
        return d

    @flx.emitter
    def submit(self):
        """ Event emitted when the user strikes the enter or return key
        (but not when losing focus). Has ``old_value`` and ``new_value``
        attributes (which are the same).
        """
        self.user_done()
        d = {'old_value': self.num, 'new_value': self.num}
        return d

    @flx.emitter
    def key_down(self, e):
        # Prevent propating the key
        ev = super().key_down(e)
        pkeys = 'Escape',  # keys to propagate
        if (ev.modifiers and ev.modifiers != ('Shift', )) or ev.key in pkeys:
            pass
        else:
            e.stopPropagation()
        if ev.key in ('Enter', 'Return'):
            self.submit()
            # Nice to blur on mobile, since it hides keyboard, but less nice on desktop
            # self.node.blur()
        elif ev.key == 'Escape':
            self.node.blur()
        return ev

    ## Reactions

    @flx.reaction
    def __num_changed(self):
        self.node.value = self.num

    @flx.reaction
    def __password_mode_changed(self):
        self.node.type = ['number', 'password'][int(bool(self.password_mode))]

    @flx.reaction
    def __placeholder_num_changed(self):
        self.node.placeholder = self.placeholder_num

    # note: this works in the browser but not in e.g. firefox-app
    @flx.reaction
    def __autocomp_changed(self):
        global window
        autocomp = self.autocomp
        # Clear
        for op in self._autocomp:
            self._autocomp.removeChild(op)
        # Add new options
        for option in autocomp:
            op = window.document.createElement('option')
            op.value = option
            self._autocomp.appendChild(op)

    @flx.reaction
    def __disabled_changed(self):
        if self.disabled:
            self.node.setAttribute("disabled", "disabled")
        else:
            self.node.removeAttribute("disabled")


class FeatureSelectorBox(flx.Widget):
    name = flx.StringProp("", settable=True)
    limits = flx.TupleProp(settable=True)
    discrete_vals = flx.ListProp([], settable=True)
    value = flx.AnyProp(settable=True)

    def init(self):
        with ui.HBox(css_class='feature-selector-container'):
            self.label = flx.Label(
                text=self.name,
                css_class='feature-selector-label',
                flex=2,
            )
            if self.discrete_vals:
                # Then we will use a combo box here to allow selection
                index = 0
                while self.discrete_vals[index] != self.value:
                    index += 1
                self.feature_selector = ui.ComboBox(
                    self.discrete_vals,
                    selected_index=index,
                    css_class='feature-selector-box',
                    flex=1,
                )
            else:
                # Then its continuous so we may use a slider or an actual
                # input box as last resource
                (low_limit, high_limit) = self.limits
                if (low_limit not in [float("inf"), -float("inf")]) and (
                    (high_limit not in [float("inf"), -float("inf")])
                ):
                    # Then we can use a slider in here!
                    with ui.VBox(css_class='slider-group', flex=1):
                        self.slider_label = flx.Label(
                            text=lambda: f'{min(max(self.value, low_limit), high_limit):.4f}',
                            css_class='slider-value-label',
                        )
                        self.feature_selector = flx.Slider(
                            min=low_limit,
                            max=high_limit,
                            value=self.value,
                            css_class="feature-selector-slider",
                        )
                else:
                    # Else we will simply use a numeric input box
                    self.feature_selector = NumberEdit(
                        num=self.value,
                        placeholder_num=str(self.value),
                        css_class="feature-selector-edit",
                        flex=1,
                    )

    @flx.reaction('!feature_selector.user_value')
    def _update_slider_value(self, *events):
        self.set_value(events[-1]['new_value'])

    @flx.reaction('!feature_selector.user_done')
    def _update_edit_value(self, *events):
        self.set_value(events[-1]['new_value'])

    @flx.reaction('!feature_selector.user_selected')
    def _update_combo_value(self, *events):
        self.set_value(events[-1]['text'])


class FeatureSelectorComponent(flx.PyWidget):
    features = flx.ListProp([], settable=True)
    feature_boxes = flx.ListProp([], settable=True)

    def init(self):
        dataset = self.root.state.dataset
        for feature in self.features:
            discrete_vals = dataset.get_allowed_values(feature)
            self._mutate_feature_boxes(
                [FeatureSelectorBox(
                    name=feature,
                    limits=dataset.get_feature_ranges(feature),
                    discrete_vals=(discrete_vals or []),
                    value=dataset.get_default_value(feature),
                )],
                'insert',
                len(self.feature_boxes),
            )

    def get_values(self):
        return [
            x.value for x in self.feature_boxes
        ]


###############################################################################
## Main Visual Component
###############################################################################

class PredictComponent(CamvizWindow):

    def init(self, ruleset):
        self.ruleset = ruleset
        self.all_features = set()
        num_used_rules_per_feat_map = defaultdict(int)

        for rule in self.ruleset.rules:
            for clause in rule.premise:
                for term in clause.terms:
                    self.all_features.add(term.variable)
                    num_used_rules_per_feat_map[term.variable] += 1

        self.all_features = list(self.all_features)
        # Make sure we display most used rules first
        self.all_features = sorted(
            self.all_features,
            key=lambda x: -num_used_rules_per_feat_map[x],
        )
        self.class_names = sorted(self.ruleset.output_class_map.keys())

        # Figure out the initial prediction for the default values
        init_vals = self._get_feature_map([
            self.root.state.dataset.get_default_value(feature)
            for feature in self.all_features
        ])
        self.predicted_val, activated_ruleset = _get_activated_ruleset(
            ruleset=self.ruleset,
            activations=init_vals,
        )
        self.confidence_level = _get_prediction_confidence(
            ruleset=activated_ruleset,
            cls_name=self.predicted_val,
        )

        with ui.HSplit(title="Prediction Explorer", flex=1):
            with ui.VSplit(flex=1) as self.prediction_pane:
                self.graph_path = PredictionPathComponent(
                    activated_ruleset,
                    predicted_val=self.predicted_val,
                    flex=0.70
                )
                with ui.GroupWidget(
                    title="Triggered Rules",
                    css_class='prediction-pane-group big-group',
                    flex=0.30,
                ):
                    self.rule_list = ClassRuleList(
                        activated_ruleset,
                        editable=False,
                    )

            with ui.VSplit(
                css_class='feature-selector',
                flex=(0.25, 1),
                style='overflow-y: scroll;',
            ):
                with ui.GroupWidget(
                    title="Predicted Result",
                    css_class='prediction-pane-group big-group',
                    flex=0.1,
                    style=(
                        "overflow-y: scroll;"
                        f"background-color: "
                        f"{self._get_color(self.predicted_val)};"
                        "text-size: 125%;"
                    ),
                ) as self.prediction_container:
                    self.prediction_label = flx.Label(
                        css_class='prediction-result',
                        html=(
                            f"{self.predicted_val} ("
                            f"confidence "
                            f"{round(self.confidence_level * 100, 2)}%)"
                        ),
                        flex=1,
                    )
                with ui.HBox(
                    css_class='feature-selector-control-panel',
                    flex=0.01
                ):
                    self.predict_button = flx.Button(
                        text="Predict",
                        css_class='predict-button',
                        flex=1,
                    )
                    self.upload_data = flx.Button(
                        text="Upload Data",
                        css_class='upload-button',
                        flex=1,
                    )
                with ui.GroupWidget(
                    title="Features",
                    css_class='scrollable_group big-group',
                    flex=0.75,
                    style="margin-bottom: 10%;",
                ):
                    self.feature_selection = FeatureSelectorComponent(
                        features=self.all_features,
                        flex=1,
                    )

    def _get_color(self, cls_name):
        cls_ind = self.root.state.ruleset.output_class_map[cls_name]
        return _CLASS_PALETTE[cls_ind % len(_CLASS_PALETTE)]

    def _get_feature_map(self, values=None):
        real_vector = {}
        values = values or self.feature_selection.get_values()
        for feature_name, val in zip(self.all_features, values):
            real_vector[feature_name] = \
                self.root.state.dataset.transform_to_numeric(feature_name, val)
        return real_vector

    @flx.emitter
    def perform_prediction(self):
        return {
            "values": self.get_values()
        }
    @flx.reaction('predict_button.pointer_click')
    def _predict_action(self, *events):
        self._act_on_prediction()

    @flx.reaction('upload_data.pointer_click')
    def _open_file_path(self, *events):
        # Set every feature according to the given file
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # And now perform a prediction
        self._act_on_prediction()

    def _act_on_prediction(self):
        real_vector = self._get_feature_map()
        self.predicted_val, activated_ruleset = _get_activated_ruleset(
            ruleset=self.ruleset,
            activations=real_vector,
        )
        self.confidence_level = _get_prediction_confidence(
            ruleset=activated_ruleset,
            cls_name=self.predicted_val,
        )
        self._update_result()
        self.graph_path.set_predicted_val(self.predicted_val)
        self.graph_path.set_ruleset(activated_ruleset)
        self.rule_list.set_ruleset(activated_ruleset)

    @flx.action
    def reset(self):
        self.ruleset = self.root.state.ruleset
        self._act_on_prediction()

    def _update_result(self):
        self.prediction_label.set_html(
            f"{self.predicted_val} (confidence "
            f"{round(self.confidence_level * 100, 2)}%)"
        )

        self.prediction_container.apply_style(
            f"background-color: {self._get_color(self.predicted_val)};"
        )
