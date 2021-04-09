from flexx import flx, ui
from gui_window import CamvizWindow
from pscript.stubs import d3, window, console
from dnn_rem.rules.ruleset import Ruleset
from dnn_rem.rules.clause import ConjunctiveClause
from dnn_rem.rules.rule import Rule
from collections import defaultdict
from rule_statistics import _CLASS_PALETTE


flx.assets.associate_asset(__name__, 'https://d3js.org/d3.v6.min.js')
flx.assets.associate_asset(
    __name__,
    'https://d3js.org/d3-scale-chromatic.v1.min.js',
)

_MAX_RADIUS = 100

_MIN_DX = 10


def _htmlify(s):
    for plain, html_code in [
        ("<=", "&leq;"),
        (">=", "&geq;"),
    ]:
        s = s.replace(plain, html_code)
    return s


def _get_term_counts(ruleset):
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
    return used_terms, num_used_rules_per_term_map


def _partition_ruleset(ruleset, term):
    contain_ruleset = Ruleset(
        rules=set(),
        feature_names=ruleset.feature_names,
        output_class_names=list(ruleset.output_class_map.keys()),
    )
    disjoint_ruleset = Ruleset(
        rules=set(),
        feature_names=ruleset.feature_names,
        output_class_names=list(ruleset.output_class_map.keys()),
    )
    for rule in ruleset.rules:
        for clause in rule.premise:
            found_term = False
            for c_term in clause.terms:
                if c_term == term:
                    found_term = True
                    break
            if found_term:
                # Then we found the term that we are looking for in this
                # rule! That means we will add it while also modifying it so
                # that the term is not included in its premise
                contain_ruleset.rules.add(Rule(
                    premise=set([
                        ConjunctiveClause(
                            terms=set(
                                [t for t in clause.terms if t != term]
                            ),
                            confidence=clause.confidence,
                            score=clause.score,
                        ),
                    ]),
                    conclusion=rule.conclusion,
                ))
            else:
                # Then add this guy into the disjoint ruleset
                disjoint_ruleset.rules.add(Rule(
                    premise=set([clause]),
                    conclusion=rule.conclusion,
                ))

    return contain_ruleset, disjoint_ruleset


def _extract_hierarchy_node(ruleset, dataset=None, merge=False):
    if not len(ruleset):
        return []
    if len(ruleset) == 1:
        # [BASE CASE]
        # Then simply output this rule (which is expected to have exactly one
        # clause)
        rule = next(iter(ruleset.rules))
        clause = next(iter(rule.premise)) if len(rule.premise) else None
        conclusion_node = {
            "name": _htmlify(str(rule.conclusion)),
            "children": [],
            "score": clause.score if clause is not None else 0,
        }
        if (clause is not None) and len(clause.terms):
            if merge:
                # Then we still have some terms left but we will not partition
                # on them as it will simply generate a chain
                return [
                    {
                        "name": _htmlify(" AND ".join(
                            map(
                                lambda x: x.to_cat_str(dataset)
                                if dataset is not None else str(x),
                                clause.terms
                            )
                        )),
                        "children": [conclusion_node],
                    },
                ]
            else:
                first = None
                current = None
                for term in clause.terms:
                    if current is None:
                        current = {
                            "name": _htmlify(
                                term.to_cat_str(dataset)
                                if dataset is not None else str(term)
                            ),
                            "children": [],
                        }
                        first = current
                    else:
                        next_elem = {
                            "name": _htmlify(
                                term.to_cat_str(dataset)
                                if dataset is not None else str(term)
                            ),
                            "children": [],
                        }
                        current["children"].append(next_elem)
                        current = next_elem
                # Finally add the conclusion
                current["children"].append(conclusion_node)
                return [first]

        # Else this is our terminal case and we add the conclusion node and
        # nothing else
        return [conclusion_node]

    # [RECURSIVE CASE]

    # Sort our nodes by the greedy metric of interest
    sorted_terms, term_count_map = _get_term_counts(
        ruleset=ruleset,
    )

    # Look at the first little bastard as this is
    # the best split in order
    next_term = sorted_terms[0]
    # Partition our ruleset around the current term
    contain_ruleset, disjoint_ruleset = _partition_ruleset(
        ruleset=ruleset,
        term=next_term,
    )

    # Construct the node for this term recursively by including it
    # in the exclude list
    next_node = {
        "name": _htmlify(
            next_term.to_cat_str(dataset)
            if dataset is not None else str(next_term)
        ),
        "children": _extract_hierarchy_node(
            ruleset=contain_ruleset,
            dataset=dataset,
            merge=merge,
        ),
    }

    # And return the result of adding this guy to our list and the
    # children resulting from the rules that do not contain it
    return [next_node] + _extract_hierarchy_node(
        ruleset=disjoint_ruleset,
        dataset=dataset,
        merge=merge,
    )


def _compute_tree_properties(tree, depth=0, merge=False):
    tree["depth"] = depth
    if len(tree["children"]) == 0:
        # Then this is a leaf!
        tree["num_descendants"] = 0
        tree["class_counts"] = {
            tree["name"]: 1,
        }
        return tree
    if (depth != 0) and len(tree["children"]) == 1 and merge and (
        len(tree["children"][0]["children"]) != 0
    ):
        # Then we can collapse this into a single node for ease of visibility
        # in this graph
        old_child = tree["children"][0]
        tree["children"] = old_child["children"]
        tree["name"] += " AND " + old_child["name"]

    # Else proceed recursively
    tree["num_descendants"] = 0
    tree["class_counts"] = {}
    class_counts = tree["class_counts"]
    for child in tree["children"]:
        child = _compute_tree_properties(child, depth=(depth + 1))
        tree["num_descendants"] += child["num_descendants"] + 1
        for class_name, count in child["class_counts"].items():
            class_counts[class_name] = count + class_counts.get(
                class_name,
                0
            )
    return tree


def ruleset_hierarchy_tree(ruleset, dataset=None, merge=False):
    tree = {
        "name": "ruleset",
        "children": _extract_hierarchy_node(
            ruleset=ruleset,
            dataset=dataset,
            merge=merge,
        ),
    }
    return _compute_tree_properties(tree, merge=merge)


def _max_depth(tree):
    result = 1
    for child in tree["children"]:
        result = max(result, _max_depth(child) + 1)
    return result


def _max_name_length(tree):
    result = len(tree["name"])
    for child in tree["children"]:
        result = max(result, _max_depth(child))
    return result


def _max_num_childs_at_depth(tree, depth):
    if depth == 0:
        if hasattr(tree, "data"):
            return tree.data.num_descendants
        return tree["num_descendants"]
    result = 0
    if hasattr(tree, "children") and (tree.children):
        for child in tree.children:
            result = max(
                result,
                _max_num_childs_at_depth(child, depth - 1)
            )
    return result


def _max_visible_depth(tree):
    if hasattr(tree, "children") and tree.children:
        result = 0
        for child in tree.children:
            result = max(result, _max_visible_depth(child) + 1)
        return result
    return 0


def _fully_expanded_depth(tree):
    # TODO: implement this
    return _max_visible_depth(tree)


def _diagonal(s, d):
    path = (
        f'M {s.y} {s.x}'
        f'C {(s.y + d.y) / 2} {s.x},'
        f'{(s.y + d.y) / 2} {d.x},'
        f'{d.y} {d.x}'
    )
    return path


class HierarchicalTreeViz(flx.Widget):
    data = flx.AnyProp(settable=True)
    fixed_node_radius = flx.FloatProp(0, settable=True)
    class_names = flx.ListProp([], settable=True)
    branch_separator = flx.FloatProp(0.2, settable=True)

    CSS = """
    .flx-HierarchicalTreeViz {
        background: #fff;
    }
    svg {
        display: block;
        margin: 0 auto;
    }

    .link {
       fill: none;
       stroke: #ccc;
       stroke-width: 2px;
    }
    """

    def init(self):
        self.node.id = self.id
        self.svg = None
        self.tooltip = None
        self.root_tree = None
        self.zoom = None

        def _startup():
            self._init_viz()
            self._load_viz()
        window.setTimeout(_startup, 500)
        window.setTimeout(lambda: self.expand_tree(), 750)

    @flx.action
    def expand_tree(self):
        self._expand_tree()

    @flx.action
    def collapse_tree(self):
        self._collapse_tree()

    @flx.action
    def clear(self):
        if self.svg:
            self.svg.selectAll("g.node").remove()
            self.svg.selectAll("path.link").remove()

    @flx.action
    def highlight_route(self, route):
        self._color_route(route=route, color="firebrick")

    @flx.action
    def unhighlight_route(self, route):
        self._color_route(route=route, color="#555")

    @flx.action
    def zoom_fit(self, duration=500):
        self._zoom_fit(duration)

    def _zoom_fit(self, duration=500):
        bounds = self.svg.node().getBBox()
        parent = self.svg.node().parentElement
        full_width = parent.clientWidth or parent.parentNode.clientWidth
        full_height = parent.clientHeight or parent.parentNode.clientHeight
        width = bounds.width
        height = bounds.height
        mid_x = bounds.x + width / 2
        mid_y = bounds.y + height / 2
        if (width == 0) or (height == 0):
            return
        scale = 0.85 / max(width / full_width, height / full_height)
        translate = [
            full_width / 2 - scale * mid_x,
            full_height / 2 - scale * mid_y
        ]

        console.trace("zoom_fit", translate, scale)

        transform = d3.zoomIdentity.translate(
            translate[0],
            translate[1],
        ).scale(
            scale
        )

        w, h = self.size
        zoom = d3.zoom().extent(
            [[0, 0], [w, h]]
        ).scaleExtent(
            [0.1, 8]
        ).on(
            "zoom",
            lambda e: self.svg.attr(
                "transform",
                e.transform,
            )
        )

        self.svg.transition().duration(
            duration
        ).call(
            zoom.transform,
            transform,
        )

    def _color_route(self, route, color="#555"):
        if not self.svg:
            # Then nothing to color in here!
            return

        # Transition links to their new color!
        self.link_update.transition().attr(
            "stroke",
            lambda d: color if (d.data.name, d.parent.data.name) in route else "#555",
        ).attr(
            "d",
            lambda d: _diagonal(d, d.parent),
        )

    @flx.reaction
    def _resize(self):
        w, h = self.size
        if len(self.node.children) > 0:
            x = d3.select('#' + self.id)
            x.attr("align", "center")
            svg = self.node.children[0]
            svg.setAttribute('width', w)
            svg.setAttribute('height', h)

            x.select("svg").select("g").attr(
                "transform",
                f'translate({w//3}, {h//2})'
            )
            graph = x.select("svg").select("g")

            def _zoomed(e):
                trans = e.transform
                graph.attr(
                    "transform",
                    f"translate({trans.x + (w//3 * trans.k)}, {trans.y + (trans.k * h//2)}) "
                    f"scale({trans.k})"

                )

            self.zoom = d3.zoom().extent(
                [[0, 0], [w, h]]
            ).scaleExtent(
                [0.1, 8]
            ).on(
                "zoom",
                _zoomed
            )
            x.select("svg").call(self.zoom)

            # And time to redraw our graph
            self._draw_graph(self.root_tree)

    def _draw_graph(
        self,
        current,
        node_size=None,
        duration=750,
        show_circles=False,
    ):
        ########################################################################
        ## Dimensions setup
        ########################################################################

        max_num_children = self.root_tree.data.num_descendants
        if self.fixed_node_radius:
            _node_radius = lambda d: self.fixed_node_radius
        else:
            _node_radius_descendants = lambda num: num/max_num_children
            _node_radius = lambda d: (
                max(
                    5,
                    _MAX_RADIUS * _node_radius_descendants(
                        d.data.num_descendants
                    )
                ) if d.data.depth else _MAX_RADIUS//2
            )
        if node_size == (0, 0):
            # Then we do not draw this just yet
            return

        self._treemap = d3.tree().size(self.size)
        if node_size:
            dx, dy = node_size
            self.root_tree.dx = dx
            self.root_tree.dy = dy
            self._treemap.nodeSize(
                [dx, dy]
            )

            def _separation_function(a, b):
                return (
                    (max(a.bounding_box[1], b.bounding_box[1])/dx) + 0.01
                    if a.parent == b.parent else self.branch_separator
                )
            self._treemap.separation(
                _separation_function
            )

        ########################################################################
        ## Construct Hierarchy Coordinates
        ########################################################################

        tree_data = self._treemap(self.root_tree)

        # Compute the new tree layout.
        nodes = tree_data.descendants()
        links = tree_data.descendants().slice(1)

        ########################################################################
        ## Draw Nodes
        ########################################################################

        def _set_node(d):
            if hasattr(d, "id") and d.id:
                return d.id
            self._id_count += 1
            d.id = self._id_count
            return d.id

        node = self.svg.selectAll("g.node").attr(
            "stroke-linejoin",
            "round"
        ).attr(
            "stroke-width",
            3,
        ).data(
            nodes,
            _set_node
        )

        # Click function for our node
        def _node_click(event, d):
            if d.children:
                # Then hide its children but save the previous children
                d._children = d.children
                d.children = None
            elif hasattr(d, "_children") and d._children:
                d.children = d._children
                d._children = None
            self._draw_graph(
                d,
                node_size,
                duration,
                show_circles,
            )

        # Create our nodes with our recursive clicking function
        node_enter = node.enter().append("g").attr(
            "class",
            "node",
        ).attr(
            "id",
            lambda d: f"node-id-{self.id}-{d.id}"
        ).attr(
            "transform",
            lambda d: f"translate({current.y0}, {current.x0})",
        ).style(
            "opacity",
            1,
        ).on(
            "click",
            _node_click,
        ).attr(
            'cursor',
            lambda d: 'pointer' if d.children or d._children else 'default',
        )

        # Use the same colors as in the other class plots for consistency
        colors = lambda i: _CLASS_PALETTE[i % len(_CLASS_PALETTE)]

        # Compute the position of each group on the pie:
        pie = d3.pie().value(
            lambda d: d.value
        )

        def arc_generator(d):
            entries = []
            idx = 0
            i = 0
            for (key, val) in d.data.class_counts.items():
                if key == d.key:
                    idx = i
                entries.append({
                    "key": key,
                    "value": val,
                })
                i += 1
            chunks = pie(entries)
            return d3.arc().innerRadius(0).outerRadius(_node_radius(d))(
                chunks[idx]
            )

        def _derp(d):
            entries = []
            total_sum = 0
            for key, val in d.data.class_counts.items():
                total_sum += val

            for key in sorted(d.data.class_counts.keys()):
                entries.append({
                    "key": key,
                    "data": d.data,
                    "percent": d.data.class_counts[key]/total_sum,
                    "children": d.children or d._children,
                })
            return entries

        _class_to_color = {}
        for i, cls_name in enumerate(self.class_names):
            _class_to_color[cls_name] = colors(i)

        pie_data = node_enter.selectAll("g").data(
            _derp
        )
        pie_enter = pie_data.enter().append("path").attr(
            "class",
            "pie_arc",
        ).attr(
            "stroke",
            "black"
        ).style(
            "stroke-width",
            "1px",
        ).style(
            "fill",
            lambda d: _class_to_color[d.key],
        ).attr(
            "d",
            arc_generator,
        ).style(
            "opacity",
            1,
        ).on(
            "mouseover",
            lambda event, d: self.tooltip.style(
                "visibility",
                "visible"
            ).html(
                (
                    f"<b>{d.key}</b>: {d.percent*100:.3f}% "
                    f"(count {d.data.class_counts[d.key]})"
                ) if d.children else f"<b>score</b>: {d.data.score}",
            )
        ).on(
            "mousemove",
            lambda event, d: self.tooltip.style(
                "top",
                f"{(event.pageY - 10)}px"
            ).style(
                "left",
                f"{(event.pageX + 10)}px",
            )
        ).on(
            "mouseout",
            lambda event, d: self.tooltip.style("visibility", "hidden")
        )

        if show_circles:
            node_enter.append("circle").attr(
                "r",
                _node_radius,
            ).style(
                "stroke",
                lambda d: 2 if d._children else 0.5,
            ).style(
                "fill",
                lambda d: colors(d.data.depth + 1),
            )

        # Add text with low opacity for now
        node_enter.append("text").attr(
            "dy",
            "0.35em"
        )

        # Transition nodes to their new position
        node_update = node_enter.merge(node)
        node_update.transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({d.y}, {d.x})",
        )

        pie_update = pie_enter.merge(pie_data)
        pie_update.on(
            "mouseover",
            lambda event, d: self.tooltip.style(
                "visibility",
                "visible"
            ).html(
                f"<b>{d.key}</b>: {d.percent*100:.3f}%"
                if d.children else f"<b>score</b>: {d.data.score}",
            )
        ).transition().duration(
            duration
        ).style(
            "fill",
            lambda d: _class_to_color[d.key],
        )

        if show_circles:
            # Show circles once transition is over
            node_update.select("circle.node").attr(
                "r",
                _node_radius,
            ).style(
                "fill",
                lambda d: colors(d.data.depth + 1),
            ).style(
                "stroke",
                lambda d: 2 if d._children else 0.5,
            )

        # And also their text
        node_update.select("text").style(
            "fill-opacity",
            1
        ).attr(
            "text-anchor",
            lambda d: "end" if d.children or d._children else "start"
        ).html(
            lambda d: d.data.name,
        ).attr(
            "font-weight",
            lambda d: "normal" if d.children or d._children else "bolder",
        ).attr(
            "font-size",
            lambda d: 10 if d.children or d._children else 19,
        ).attr(
            "x",
            lambda d: (
                -(_node_radius(d) + 3) if d.children or d._children
                else (_node_radius(d) + 3)
            ),
        ).attr(
            "fill",
            lambda d: (
                "black" if d.children or d._children
                else _class_to_color[d.data.name]
            )
        )

        if node_size is None:
            def _compute_bounding_box(d, i):
                bbox = d3.select(f"#node-id-{self.id}-{d.id}").node().getBBox()
                d["bounding_box"] = (bbox.width, bbox.height)
            node_update.each(_compute_bounding_box)
            dx = d3.max(nodes, lambda d: d.bounding_box[1])
            dy = d3.max(nodes, lambda d: d.bounding_box[0])
            # And re-run this whole thing with our corrected bounding box values
            self._draw_graph(
                current,
                [dx, dy],
                duration,
                show_circles,
            )
            return

        # Transition function for the node exit
        # First remove our node
        node_exit = node.exit().transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({current.y}, {current.x})",
        ).remove()

        if show_circles:
            # Then make the circle transparent
            node_exit.select("circle").attr(
                "r",
                1e-6,
            )

        # Then make the circle transparent
        node_exit.select("g.path").attr(
            "opacity",
            1e-6,
        )

        # And the text as well
        node_exit.select("text").style(
            "fill-opacity",
            1e-6,
        )

        ########################################################################
        ## Draw Links
        ########################################################################
        link = self.svg.selectAll("path.link").data(
            links,
            lambda d: d.id,
        )

        link_enter = link.enter().insert(
            "path",
            "g"
        ).attr(
            "class",
            "link",
        ).attr(
            "d",
            # Links for now will be kept invisible by remaining in the
            # same spot
            lambda d: _diagonal(
                {"x": current.x0, "y": current.y0},
                {"x": current.x0, "y": current.y0},
            )
        ).attr(
            "fill",
            "none",
        ).attr(
            "stroke",
            lambda d: "#555",
        ).attr(
            "stroke-opacity",
            lambda d: 1,
        ).attr(
            "stroke-width",
            lambda d: 15,
        )

        # Transition links to their new position when the clicking happens
        self.link_update = link_enter.merge(link)
        self.link_update.transition().duration(
            duration
        ).attr(
            "d",
            lambda d: _diagonal(d, d.parent),
        )

        # And put back exit links into hiding
        link.exit().transition().duration(
            duration
        ).attr(
            "d",
            lambda d: _diagonal(
                {"x": current.x, "y": current.y},
                {"x": current.x, "y": current.y},
            )
        ).remove()

        # Finally, save old positions so that we can transition next
        def _save_positions(d):
            d.x0 = d.x
            d.y0 = d.y
        nodes.forEach(_save_positions)


    @flx.reaction('data')
    def reload_viz(self, *events):
        if self.svg:
            self.svg.selectAll("g.node").remove()
            self.svg.selectAll("path.link").remove()
            self._load_viz()

    def _init_viz(self):
        x = d3.select('#' + self.id)
        width, height = self.size
        width = max(width, 600)
        height = max(height, 600)
        self.svg = x.append("svg").attr(
            "width",
            width
        ).attr(
            "height",
            height
        ).append(
            "g"
        ).attr(
            "transform",
            f"translate({width//3}, {height//3})"
        )
        x.attr("align", "center")

        # Generate a tooltip for displaying different messages
        self.tooltip = d3.select("body").append("div").style(
            "position",
            "absolute",
        ).style(
            "z-index",
            "10",
        ).style(
            "visibility",
            "hidden",
        ).text("")

    def _load_viz(self, expand=False):
        self._id_count = 0
        _, height = self.size
        self.root_tree = d3.hierarchy(
            self.data,
            lambda d: d.children,
        )
        self.root_tree.x0 = height/2
        self.root_tree.y0 = 0
        self._expand_tree()

    def _collapse_tree(self, root_too=False):
        def _collapse(d):
            if d.children:
                d._children = d.children
                d._children.forEach(_collapse)
                d.children = None
            elif hasattr(d, "_children") and d._children:
                # Then make sure we collapse all inner children here as well
                # in case things have been partially collapsed
                d._children.forEach(_collapse)
        if root_too:
            _collapse(self.root_tree)
        elif self.root_tree.children:
            self.root_tree.children.forEach(_collapse)
        self._draw_graph(self.root_tree)

    def _expand_tree(self):
        def _expand_node(d):
            if hasattr(d, "_children") and (d._children):
                d.children = d._children
                d._children = None
            if d.children:
                d.children.forEach(_expand_node)
        if not self.root_tree.children:
            if (
                hasattr(self.root_tree, "_children") and
                (self.root_tree._children is not None)
            ):
                self.root_tree.children = self.root_tree._children
                self.root_tree._children = None
        if self.root_tree.children:
            self.root_tree.children.forEach(_expand_node)
        self._draw_graph(self.root_tree)


class RuleExplorerComponent(CamvizWindow):

    def _compute_hierarchical_tree(self):
        return ruleset_hierarchy_tree(
            ruleset=self.root.state.ruleset,
            dataset=self.root.state.dataset,
            merge=self.root.state.merge_branches,
        )

    def init(self):
        with ui.VSplit(
            title="Rule Explorer",
        ):
            self.tree = HierarchicalTreeViz(
                data=self._compute_hierarchical_tree(),
                class_names=self.root.state.ruleset.output_class_names(),
                flex=0.95,
            )
            with ui.HBox(0.05):
                ui.Widget(flex=1)
                self.expand_button = flx.Button(
                    text='Expand Tree',
                    flex=0,
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.collapse_button = flx.Button(
                    text='Collapse Tree',
                    flex=0,
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=0.25)
                self.fit_button = flx.Button(
                    text="Fit to Screen",
                    css_class='tool-bar-button',
                )
                ui.Widget(flex=1)
            ui.Widget(flex=0.05)

    @flx.reaction('expand_button.pointer_click')
    def _expand_clicked(self, *events):
        self.tree.expand_tree()
        self.tree.zoom_fit()

    @flx.reaction('collapse_button.pointer_click')
    def _collapse_clicked(self, *events):
        self.tree.collapse_tree()
        self.tree.zoom_fit()

    @flx.reaction('fit_button.pointer_click')
    def _fit_clicked(self, *events):
        self.tree.zoom_fit()

    @flx.action
    def reset(self):
        self.tree.set_data(self._compute_hierarchical_tree())
