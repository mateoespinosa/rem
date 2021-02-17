import bokeh
import numpy as np

from flexx import flx, ui
from collections import defaultdict
from gui_window import CamvizWindow
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.palettes import Category20, Category20c, Set3
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models.tickers import AdaptiveTicker
from bokeh.transform import cumsum

################################################################################
## Global Variables
################################################################################

_PLOT_WIDTH = 750
_PLOT_HEIGHT = 360

################################################################################
## Helper Plot Constructors
################################################################################


def _plot_rule_distribution(ruleset, show_tools=True, add_labels=False):
    rules_per_class_map = defaultdict(int)
    for rule in ruleset.rules:
        for clause in rule.premise:
            rules_per_class_map[rule.conclusion] += 1

    output_classes = list(ruleset.output_class_map.keys())
    rules_per_class = [
        rules_per_class_map[cls_name] for cls_name in output_classes
    ]
    total_rules = sum(rules_per_class)
    source = ColumnDataSource(
        data={
            "Output Classes": output_classes,
            "Number of Rules": rules_per_class,
            'Angle': [
                num_rules/total_rules * 2*np.pi
                for num_rules in rules_per_class
            ],
            'Percent': [
                f'{100 * num_rules/total_rules:.2f}%'
                for num_rules in rules_per_class
            ],
        }
    )
    result_plot = figure(
        # x_range=(-0.5, 1.0),
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title=f"Rule counts per class (total number of rules {total_rules})",
    )
    result_plot.annular_wedge(
        x=0,
        y=1,
        inner_radius=0.2,
        outer_radius=0.4,
        start_angle=cumsum('Angle', include_zero=True),
        end_angle=cumsum('Angle'),
        line_color="white",
        fill_color=factor_cmap(
            'Output Classes',
            palette=Set3[12],
            factors=output_classes,
        ),
        legend_field='Output Classes',
        source=source,
    )
    if add_labels:
        percent_labels = LabelSet(
            x=0,
            y=1,
            text='Percent',
            angle=cumsum('Angle', include_zero=True),
            source=source,
            render_mode='canvas',
        )
        result_plot.add_layout(percent_labels)

    result_plot.axis.axis_label = None
    result_plot.axis.visible = False
    result_plot.grid.grid_line_color = None
    result_plot.legend.location = "center_right"

    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    hover = HoverTool(tooltips=[
        ('Class', '@{Output Classes}'),
        ('Count', '@{Number of Rules}'),
        ('Percent', '@{Percent}'),
    ])
    result_plot.add_tools(hover)
    return result_plot


def _plot_term_distribution(
    ruleset,
    show_tools=True,
    max_entries=float("inf"),
):
    num_used_rules_per_term_map = defaultdict(int)
    all_terms = set()
    for rule in ruleset.rules:
        for clause in rule.premise:
            for term in clause.terms:
                all_terms.add(term)
                num_used_rules_per_term_map[term] += 1

    all_terms = list(all_terms)
    # Make sure we display most used rules first
    used_terms = sorted(
        all_terms,
        key=lambda x: -num_used_rules_per_term_map[x],
    )
    if max_entries != float("inf"):
        used_terms = used_terms[:max_entries]
    # And we will pick only the requested top entries
    num_used_rules_per_term = [
        num_used_rules_per_term_map[term] for term in used_terms
    ]
    used_terms = list(map(str, used_terms))
    source = ColumnDataSource(
        data={
            "Terms": used_terms,
            "Rules Using that Term": num_used_rules_per_term,
        }
    )
    title = f"Top {min(max_entries, len(used_terms))} used terms"
    if len(used_terms) != len(all_terms):
        title += (
            f" (out of {len(all_terms)} unique terms used in all the ruleset)"
        )
    result_plot = figure(
        x_range=used_terms,
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title=title,
    )
    result_plot.vbar(
        x='Terms',
        top='Rules Using that Term',
        width=0.9,
        source=source,
        line_color='white',
        fill_color=factor_cmap(
            'Terms',
            palette=Category20c[20],
            factors=used_terms,
        ),
    )
    result_plot.xgrid.grid_line_color = None
    result_plot.y_range.start = 0
    result_plot.y_range.end = int(max(0, 0, *num_used_rules_per_term) * 1.1)
    result_plot.xaxis.major_label_orientation = 1.15
    result_plot.xaxis.axis_label = "Terms"
    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    hover = HoverTool(tooltips=[
        ('Count', '@{Rules Using that Term}'),
        ('Term', '@{Terms}'),
    ])
    result_plot.add_tools(hover)
    return result_plot


def _plot_feature_distribution(
    ruleset,
    show_tools=True,
    max_entries=float("inf"),
):
    num_used_rules_per_feat_map = defaultdict(int)
    all_features = set()
    for rule in ruleset.rules:
        for clause in rule.premise:
            for term in clause.terms:
                all_features.add(term.variable)
                num_used_rules_per_feat_map[term.variable] += 1

    all_features = list(all_features)
    # Make sure we display most used rules first
    used_features = sorted(
        all_features,
        key=lambda x: -num_used_rules_per_feat_map[x],
    )
    if max_entries != float("inf"):
        used_features = used_features[:max_entries]
    # And we will pick only the requested top entries
    num_used_rules_per_feat = [
        num_used_rules_per_feat_map[term] for term in used_features
    ]
    used_features = list(map(str, used_features))
    source = ColumnDataSource(
        data={
            "Feature": used_features,
            "Rules Using that Feature": num_used_rules_per_feat,
        }
    )
    title = f"Top {min(max_entries, len(used_features))} used features"
    if len(used_features) != len(all_features):
        title += (
            f" (out of {len(all_features)}/{len(ruleset.feature_names)} "
            f"features used in all the ruleset)"
        )
    result_plot = figure(
        x_range=used_features,
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title=title,
    )
    result_plot.vbar(
        x='Feature',
        top='Rules Using that Feature',
        width=0.9,
        source=source,
        line_color='white',
        fill_color=factor_cmap(
            'Feature',
            palette=Category20[20],
            factors=used_features,
        ),
    )
    result_plot.xgrid.grid_line_color = None
    result_plot.y_range.start = 0
    result_plot.y_range.end = int(max(0, 0, *num_used_rules_per_feat) * 1.1)
    result_plot.xaxis.major_label_orientation = 1.15
    result_plot.xaxis.axis_label = "Feature"
    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    hover = HoverTool(tooltips=[
        ('Count', '@{Rules Using that Feature}'),
        ('Feature', '@{Feature}'),
    ])
    result_plot.add_tools(hover)
    return result_plot


def _plot_rule_length_distribution(
    ruleset,
    show_tools=True,
    max_entries=float("inf"),
    num_bins=10,
):
    class_rule_lengths = [
        [] for _ in ruleset.output_class_map
    ]
    output_classes = [
        cls_name for cls_name in ruleset.output_class_map.keys()
    ]
    output_classes.sort(key=lambda x: ruleset.output_class_map[x])
    for rule in ruleset.rules:
        for clause in rule.premise:
            class_rule_lengths[
                ruleset.output_class_map[rule.conclusion]
            ].append(len(clause.terms))

    palette = Set3[12]
    result_plot = figure(
        toolbar_location=None if (not show_tools) else "right",
        plot_width=_PLOT_WIDTH,
        plot_height=_PLOT_HEIGHT,
        background_fill_color="#fafafa",
        title="Rule length distribution (click legend to hide classes)",
    )
    for cls_name, rule_lengths in zip(output_classes, class_rule_lengths):
        hist, edges = np.histogram(
            rule_lengths,
            bins=min(num_bins, len(rule_lengths)),
        )
        result_plot.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color=palette[ruleset.output_class_map[cls_name]],
            line_color="black",
            alpha=0.5,
            legend_label=cls_name,
        )
    result_plot.y_range.start = 0
    result_plot.legend.location = "center_right"
    result_plot.legend.background_fill_color = "#fefefe"
    result_plot.xgrid.grid_line_color = None
    result_plot.xaxis.axis_label = 'Rule Length'
    result_plot.toolbar.logo = None
    for tool in result_plot.toolbar.tools:
        if isinstance(
            tool,
            (bokeh.models.tools.HelpTool)
        ):
            result_plot.toolbar.tools.remove(tool)
    result_plot.yaxis.axis_label = 'Count'
    result_plot.legend.click_policy = "hide"
    hover = HoverTool(tooltips=[
        ('Count', '@top'),
        ('Range', '(@left, @right)'),
    ])
    result_plot.add_tools(hover)
    return result_plot


################################################################################
## Main Widget Class
################################################################################


class RuleStatisticsComponent(CamvizWindow):
    groups = flx.ListProp(settable=True)
    rows = flx.ListProp(settable=True)

    def init(self):
        self.ruleset = self.root.state.ruleset
        self.show_tools = self.root.state.show_tools
        self.max_entries = self.root.state.max_entries
        with ui.VSplit(
            title="Ruleset Summary",
            style=(
                'overflow-y: scroll;'
                'overflow-x: scroll;'
            )
        ) as self.container:
            self._mutate_rows(
                [
                    ui.HBox(
                        flex=1,
                        style=(
                            'overflow-y: scroll;'
                            'overflow-x: scroll;'
                        ),
                    )
                ],
                'insert',
                len(self.rows)
            )
            self._mutate_rows(
                [
                    ui.HBox(
                        flex=1,
                        style=(
                            'overflow-y: scroll;'
                            'overflow-x: scroll;'
                        ),
                    )
                ],
                'insert',
                len(self.rows)
            )
        self._construct_plots()

    @flx.action
    def add_plot(self, title, plot):
        with ui.Widget(
            title=title,
            style=(
                'overflow-y: scroll;'
                'overflow-x: scroll;'
            ),
            flex=1,
        ) as new_group:
            new_plot = ui.BokehWidget.from_plot(plot, flex=1)
            self._mutate_groups(
                [new_group],
                'insert',
                len(self.groups),
            )

    @flx.action
    def _construct_plots(self):
        with self.container:
            with self.rows[0]:
                self.add_plot(
                    "Rule Distribution",
                    _plot_rule_distribution(
                        ruleset=self.ruleset,
                        show_tools=self.show_tools,
                    ),
                )
                self.add_plot(
                    "Rule Length Distribution",
                    _plot_rule_length_distribution(
                        ruleset=self.ruleset,
                        show_tools=self.show_tools,
                        max_entries=self.max_entries,
                    ),
                )
            with self.rows[1]:
                self.add_plot(
                    "Feature Distribution",
                    _plot_feature_distribution(
                        ruleset=self.ruleset,
                        show_tools=self.show_tools,
                        max_entries=self.max_entries,
                    ),
                )
                self.add_plot(
                    "Term Distribution",
                    _plot_term_distribution(
                        ruleset=self.ruleset,
                        show_tools=self.show_tools,
                        max_entries=self.max_entries,
                    ),
                )

    @flx.action
    def reset(self):
        for group in self.groups:
            group.set_parent(None)
        for row in self.rows:
            row.set_parent(None)

        self._mutate_groups([])
        self._mutate_rows([])
        with self:
            with self.container:
                self._mutate_rows(
                    [
                        ui.HBox(
                            flex=1,
                            style=(
                                'overflow-y: scroll;'
                                'overflow-x: scroll;'
                            ),
                        )
                    ],
                    'insert',
                    len(self.rows)
                )
                self._mutate_rows(
                    [
                        ui.HBox(
                            flex=1,
                            style=(
                                'overflow-y: scroll;'
                                'overflow-x: scroll;'
                            ),
                        )
                    ],
                    'insert',
                    len(self.rows)
                )
            self._construct_plots()
