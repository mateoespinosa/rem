from flexx import flx, ui
from gui_window import CamvizWindow
from collections import defaultdict
from pscript.stubs import d3, window, Math, Set
from dnn_rem.rules.term import TermOperator
from pscript import RawJS
from sklearn.neighbors import KernelDensity
import numpy as np
from rule_statistics import _CLASS_PALETTE

def _kernel_density_estimator(kernel, x_points):
    def _result_fn(values):
        return x_points.map(
            lambda x: [x, d3.mean(values, lambda v: kernel(x - v))],
        )
    return _result_fn


def _kernel_epanechnikov(k):
    def _result_fn(v):
        v /= k
        if abs(v) <= 1:
            return 0.75 * (1 - v * v) / k
        return 0
    return _result_fn


class FeatureBoundView(flx.Widget):
    CSS = """
        path {
            fill: none;
            stroke: #aaa;
        }
    """
    DEFAULT_MIN_SIZE = 800, 500

    feature_name = flx.StringProp(settable=True)
    feature_limits = flx.TupleProp(settable=True)
    rule_bounds = flx.DictProp(settable=True)
    num_ticks = flx.IntProp(20, settable=True)
    data = flx.ListProp([], settable=True)
    classes = flx.ListProp([], settable=True)
    estimated_densities = flx.DictProp({}, settable=True)
    plot_density = flx.BoolProp(False, settable=True)

    def init(self):
        self.node.id = self.id
        window.setTimeout(self.load_viz, 500)

    @flx.action
    def load_viz(self):
        width, height = self.DEFAULT_MIN_SIZE
        left_margin, right_margin = 50, 50
        top_margin, bottom_margin = 50, 50

        x = d3.select('#' + self.id)
        top_div = x.append("div").attr(
            "class",
            "select-container",
        )
        top_div.append("div").attr(
            "class",
            "select-label",
        ).html(
            "Class"
        )
        self.class_select = top_div.append("select").attr(
            "class",
            "class-select",
        )

        self.svg = x.append("svg").attr(
            "width",
            width
        ).attr(
            "height",
            height,
        ).attr(
            "xmlns",
            "http://www.w3.org/2000/svg",
        ).attr(
            "version",
            "1.1"
        )

        ########################################################################
        ## Attach pattern definitions
        ########################################################################

        self.svg.append(
            "defs",
        ).append(
            "pattern",
        ).attr(
            "id",
            "crosshatch",
        ).attr(
            "patternUnits",
            "userSpaceOnUse",
        ).attr(
            "width",
            8,
        ).attr(
            "height",
            8,
        ).append(
            "img"
        ).attr(
            "xlink:href",
            "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc4JyBoZWlnaHQ9JzgnPgogIDxyZWN0IHdpZHRoPSc4JyBoZWlnaHQ9JzgnIGZpbGw9JyNmZmYnLz4KICA8cGF0aCBkPSdNMCAwTDggOFpNOCAwTDAgOFonIHN0cm9rZS13aWR0aD0nMC41JyBzdHJva2U9JyNhYWEnLz4KPC9zdmc+Cg==",
        ).attr(
            "x",
            0,
        ).attr(
            "y",
            0,
        ).attr(
            "width",
            8,
        ).attr(
            "height",
            8,
        )

        self.svg.append(
            'defs'
        ).append(
            'pattern'
        ).attr(
            'id',
            'diagonalHatch',
        ).attr(
            'patternUnits',
            'userSpaceOnUse',
        ).attr(
            'width',
            4,
        ).attr(
            'height',
            4,
        ).append(
            'path'
        ).attr(
            'd',
            'M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2',
        ).attr(
            'stroke',
            '#000000',
        ).attr(
            'stroke-width',
            1
        )

        ########################################################################
        ## Populate the class selection
        ########################################################################

        x.select(".class-select").selectAll(
            'myOptions'
        ).data(
            self.classes
        ).enter(
        ).append(
            'option'
        ).text(
            lambda d: d,
        ).attr(
            "value",
            lambda d: d,
        )

        ########################################################################
        ## Add x-axis
        ########################################################################

        x_scale = d3.scaleLinear().domain(
            [(x * 1.0) for x in self.feature_limits]
        ).range(
            [left_margin, width - right_margin]
        )
        x_axis_y_position = height - bottom_margin - 50
        x_axis = self.svg.append(
            "g"
        ).attr(
            "class",
            "axis",
        ).attr(
            "transform",
            f"translate({0}, {x_axis_y_position})"
        ).call(
            d3.axisBottom(x_scale).ticks(min(max(2, width // 20), 25))
        )

        x_axis.selectAll("path").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            3,
        ).attr(
            "stroke-linecap",
            "round"
        )

        x_axis.selectAll(".tick").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            1,
        )

        x_axis.selectAll(".tick:first-of-type").attr(
            "stroke",
            "red",
        ).attr(
            "stroke-width",
            2,
        )
        x_axis.selectAll(".tick:last-of-type").attr(
            "stroke",
            "red",
        ).attr(
            "stroke-width",
            2,
        )

        label_x = (width - right_margin - left_margin) // 2 + left_margin
        x_label = x_axis.append(
            "text"
        ).attr(
            "x",
            label_x,
        ).attr(
            "y",
            45,
        ).attr(
            "text-anchor",
            "middle",
        ).attr(
            "fill",
            "currentColor"
        ).text(
            self.feature_name
        ).attr(
            "class",
            "axis-label",
        )

        x_label.selectAll(
            "text"
        ).clone(
            True
        ).lower().attr(
            "fill",
            "none",
        ).attr(
            "stroke-width",
            5,
        ).attr(
            "stroke-linejoin",
            "round",
        ).attr(
            "stroke",
            "white",
        )

        histogram = d3.histogram().value(
            lambda d: d[self.feature_name]
        ).domain(
            x_scale.domain()
        ).thresholds(
            x_scale.ticks(50)
        )
        self.class_bins = {}
        max_y = 0
        for cls_name in self.classes:
            self.class_bins[cls_name] = histogram(list(filter(
                lambda d: d["class"] == cls_name,
                self.data
            )))
            normalize_factor = d3.sum(
                self.class_bins[cls_name],
                lambda d: d.length
            )
            for bin_d in self.class_bins[cls_name]:
                bin_d.norm_val = bin_d.length / normalize_factor
            max_y = max(
                max_y,
                d3.max(self.class_bins[cls_name], lambda d: d.norm_val)
            ),

        ########################################################################
        ## Add y-axis
        ########################################################################

        y_scale = d3.scaleLinear().domain(
            [0, max_y]
        ).range(
            [x_axis_y_position, top_margin]
        )
        y_axis = self.svg.append(
            "g"
        ).attr(
            "class",
            "axis",
        ).attr(
            "transform",
            f"translate({left_margin}, {0})"
        ).call(
            d3.axisLeft(y_scale).ticks(min(max(2, height // 20), 25))
        )
        y_axis.selectAll("path").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            3,
        ).attr(
            "stroke-linecap",
            "round"
        )
        y_axis.selectAll(".tick").attr(
            "stroke",
            "black",
        ).attr(
            "stroke-width",
            1,
        )

        ########################################################################
        ## Empirical Distribution
        ########################################################################

        empirical_dist = self.svg.selectAll(
            "rect",
        ).data(
            self.class_bins[self.classes[0]],
        ).enter(
        ).append(
            "rect",
        ).attr(
            "x",
            1,
        ).attr(
            "transform",
            lambda d: f"translate({x_scale(d.x0)}, {y_scale(d.norm_val)})"
        ).attr(
            "width",
            lambda d: max(x_scale(d.x1) - x_scale(d.x0) - 1, 0),
        ).attr(
            "height",
            lambda d: height - top_margin - bottom_margin - y_scale(d.norm_val),
        ).style(
            "fill",
            _CLASS_PALETTE[0],
        ).attr(
            "opacity",
            0.5,
        )

        ########################################################################
        ## Plot data density estimation
        ########################################################################

        # We default to the first class to construct the distribution
        if self.plot_density:
            density = self.estimated_densities[self.classes[0]]

            # Time to make a pretty area diagram here!
            distribution_curve = self.svg.append(
                'g'
            ).append(
                "path"
            ).attr(
                "class",
                "mypath",
            ).datum(
                density
            ).attr(
                "fill",
                "#69b3a2",
            ).attr(
                "opacity",
                0.8,
            ).attr(
                "stroke",
                "black",
            ).attr(
                "stroke-width",
                3,
            ).attr(
                "stroke-linejoin",
                "round",
            ).attr(
                "d",
                d3.line().curve(
                    d3.curveBasis
                ).x(
                    lambda d: x_scale(d[0])
                ).y(
                    lambda d: y_scale(
                        d[1]
                    )
                )
            )

        ########################################################################
        ## Plot thresholds
        ########################################################################
        interval_group = self.svg.append("g").attr("class", "interval-group")
        interval_group.selectAll("rect").data(
            self.rule_bounds.get(self.classes[0], []),
        ).enter(
        ).append(
            "rect"
        ).attr(
            "class",
            "threshold-shading"
        ).attr(
            "x",
            lambda d: x_scale(d[0])
        ).attr(
            "y",
            y_scale(max_y)
        ).attr(
            "width",
            lambda d: x_scale(d[1]) - x_scale(d[0]),
        ).attr(
            "height",
            x_axis_y_position - top_margin
        ).attr(
            "opacity",
            0.25
        ).attr(
            "fill",
            "#e400ff",  # "url(#crosshatch)",
        ).attr(
            "stroke",
            "red",
        ).attr(
            "stroke-width",
            4,
        ).attr(
            'stroke-dasharray',
            '1,12',
        ).attr(
            'stroke-linecap',
            'square',
        )

        ########################################################################
        ## Update Function
        ########################################################################

        # A function to call whenever the selected class changes
        def _update_plot(new_class):
            # Recompute the density here
            bins = self.class_bins[new_class]
            if self.plot_density:
                density = self.estimated_densities[new_class]

                distribution_curve.datum(
                    density,
                ).transition(
                ).duration(
                    1000,
                ).attr(
                    "d",
                    d3.line().curve(
                        d3.curveBasis
                    ).x(
                        lambda d: x_scale(d[0])
                    ).y(
                        lambda d: y_scale(d[1]),
                    )
                )

            # And the empirical distribution as well
            empirical_dist.data(
                bins,
            ).transition(
            ).duration(
                1000,
            ).attr(
                "transform",
                lambda d: f"translate({x_scale(d.x0)}, {y_scale(d.norm_val)})"
            ).attr(
                "width",
                lambda d: max(x_scale(d.x1) - x_scale(d.x0) - 1, 0)
            ).attr(
                "height",
                lambda d: height - top_margin - bottom_margin - y_scale(
                    d.norm_val
                )
            ).style(
                "fill",
                _CLASS_PALETTE[
                    self.classes.index(new_class) % len(_CLASS_PALETTE)
                ],
            )

            # Finally, update the threshold
            intervals = interval_group.selectAll("rect").data(
                self.rule_bounds.get(new_class, []),
            )

            interval_enter = intervals.enter().append(
                "rect"
            ).attr(
                "class",
                "threshold-shading"
            ).attr(
                "x",
                # Have it "grow" from the middle of the interval by first
                # placing it in the center and then moving it left while we also
                # increase its width
                lambda d: x_scale(d[0]) + (x_scale(d[1]) - x_scale(d[0]))/2,
            ).attr(
                "y",
                y_scale(max_y)
            ).attr(
                "width",
                lambda d: 0,
            ).attr(
                "height",
                lambda d: x_axis_y_position - top_margin,
            ).attr(
                "opacity",
                0.25
            ).attr(
                "fill",
                "#e400ff",  # "url(#crosshatch)",
            ).attr(
                "stroke",
                "red",
            ).attr(
                "stroke-width",
                4,
            ).attr(
                'stroke-dasharray',
                '1,12',
            ).attr(
                'stroke-linecap',
                'square',
            )

            interval_update = interval_enter.merge(intervals)
            interval_update.transition().duration(
                1000,
            ).attr(
                "x",
                lambda d: x_scale(d[0]),
            ).attr(
                "width",
                lambda d: x_scale(d[1]) - x_scale(d[0]),
            )

            interval_exit = intervals.exit().transition(
            ).duration(
                1000,
            ).attr(
                "x",
                lambda d: x_scale(d[0]) + (x_scale(d[1]) - x_scale(d[0]))/2,
            ).attr(
                "width",
                lambda d: 0,
            ).remove()

        # And make sure our graph gets updated whenever the class selection
        # is changed
        def _on_change_selection(sel):
            global document
            x = document.getElementById(f'{self.id}')
            return _update_plot(x.getElementsByTagName("select")[0].value)

        x.select(".class-select").on(
            "change",
            _on_change_selection,
        )


def _collapse_intervals(intervals):
    result = []
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    current_low, current_high = intervals[0]
    for (next_low, next_high) in intervals:
        if next_low <= current_high:
            # Then merge these two!
            current_high = max(current_high, next_high)
        else:
            # Else time to add the current interval to our list and move on
            # to form the following one
            result.append((current_low, current_high))
            current_low, current_high = next_low, next_high

    # And we need to add the current interval here
    result.append((current_low, current_high))
    return result


class FeatureBoundComponent(flx.PyWidget):
    feature = flx.AnyProp(settable=True)
    feature_limits = flx.TupleProp(settable=True)

    def init(self, feature, feature_limits):
        self._mutate_feature(feature)
        self._mutate_feature_limits(feature_limits)
        ruleset = self.root.state.ruleset
        interval_map = defaultdict(list)
        for rule in ruleset:
            for clause in rule.premise:
                for term in clause.terms:
                    if term.variable != self.feature:
                        continue
                    # Then include this interval in its corresponding class
                    if term.operator == TermOperator.GreaterThan:
                        bound = (term.threshold, self.feature_limits[1])
                    else:
                        bound = (self.feature_limits[0], term.threshold)
                    interval_map[rule.conclusion].append(bound)
        # Time to collapse intervals
        for cls_name, intervals in interval_map.items():
            interval_map[cls_name] = _collapse_intervals(intervals)
        data = []
        dataset = self.root.state.dataset
        inv_name_map = {}
        for (cls_name, cls_code) in ruleset.output_class_map.items():
            inv_name_map[cls_code] = cls_name

        # Do some min-max scaling. For that we need to find the max and the
        # min value of the dataset here
        max_val = max(dataset[self.feature])
        min_val = min(dataset[self.feature])
        for (val, cls_name) in zip(
            dataset[self.feature],
            dataset[dataset.columns[-1]]
        ):
            scaled_val = (val - min_val) / (max_val - min_val)
            data.append({
                "class": inv_name_map[cls_name],
                self.feature: scaled_val,
            })

        estimated_densities = {}
        classes = list(ruleset.output_class_map.keys())
        for cls_name in classes:
            values = list(map(
                lambda x: x[self.feature],
                filter(
                    lambda x: x["class"] == cls_name,
                    data
                )
            ))
            values = np.array(values).reshape(-1, 1)
            kernel = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(
                values
            )
            x_vals = np.linspace(0, 1, 200).reshape(-1, 1)
            density = np.exp(kernel.score_samples(x_vals))
            estimated_densities[cls_name] = list(zip(
                x_vals.flatten(),
                density.flatten(),
            ))

        FeatureBoundView(
            feature_name=self.feature,
            feature_limits=self.feature_limits,
            rule_bounds=dict(interval_map),
            data=data,
            classes=classes,
            estimated_densities=estimated_densities,
        )


class FeatureExplorerComponent(CamvizWindow):

    feature_views = flx.ListProp(settable=True)
    current_window = flx.IntProp(0, settable=True)

    def init(self):
        self.ruleset = self.root.state.ruleset
        self.all_features = set()
        num_used_rules_per_feat_map = defaultdict(int)
        for rule in self.ruleset.rules:
            for clause in rule.premise:
                for term in clause.terms:
                    self.all_features.add(term.variable)
                    num_used_rules_per_feat_map[term] += 1

        self.all_features = list(self.all_features)
        # Make sure we display most used rules first
        self.all_features = sorted(
            self.all_features,
            key=lambda x: -num_used_rules_per_feat_map[x],
        )
        self.class_names = sorted(self.ruleset.output_class_map.keys())
        with ui.VBox(
            title="Feature Explorer",
        ):
            with ui.StackLayout(flex=1) as self.stack:
                for feature in self.all_features:
                    self._mutate_feature_views(
                        # TODO: bounds should come from dataset rather than
                        # assumed to be [0, 1]
                        [FeatureBoundComponent(feature, (0.0, 1.0))],
                        'insert',
                        len(self.feature_views),
                    )
            with ui.HBox() as self.control_panel:
                ui.Widget(flex=1)  # filler
                self.feature_selection = ui.ComboBox(
                    options=self.all_features,
                    selected_index=0,
                    css_class='feature-selection-box',
                )
                ui.Widget(flex=1)  # filler
            ui.Widget(flex=0.25)  # filler

    @flx.action
    def select_feature(self, feature_ind):
        self.set_current_window(feature_ind)
        self.stack.set_current(self.current_window)

    @flx.reaction('feature_selection.user_selected')
    def perform_selection(self, *events):
        self.select_feature(events[-1]['index'])

    @flx.action
    def reset(self):
        # TODO
        pass
