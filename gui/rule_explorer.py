from flexx import flx, ui
from gui_window import CamvizWindow
from pscript import RawJS
from pscript.stubs import Math, d3, window
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

_MAX_RADIUS = 150

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


def _extract_hierarchy_node(ruleset):
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
            # Then we still have some terms left but we will not partition
            # on them as it will simply generate a chain
            return [
                {
                    "name": _htmlify(" <b>AND</b> ".join(
                        map(str, clause.terms))
                    ),
                    "children": [conclusion_node],
                },
            ]
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
        "name": _htmlify(str(next_term)),
        "children": _extract_hierarchy_node(ruleset=contain_ruleset),
    }

    # And return the result of adding this guy to our list and the
    # children resulting from the rules that do not contain it
    return [next_node] + _extract_hierarchy_node(ruleset=disjoint_ruleset)


def _compute_tree_properties(tree, depth=0):
    tree["depth"] = depth
    if len(tree["children"]) == 0:
        # Then this is a leaf!
        tree["num_descendants"] = 0
        tree["class_counts"] = {
            tree["name"]: 1,
        }
        return tree
    if len(tree["children"]) == 1 and (
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


def _ruleset_hierarchy_tree(ruleset):
    tree = {
        "name": "ruleset",
        "children": _extract_hierarchy_node(ruleset=ruleset),
    }
    return _compute_tree_properties(tree)


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
        window.setTimeout(self.load_viz, 500)

    @flx.action
    def expand_tree(self):
        self._expand_tree()

    @flx.action
    def collapse_tree(self):
        self._collapse_tree()

    @flx.reaction
    def _resize(self):
        w, h = self.parent.size
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

            x.select("svg").call(
                d3.zoom().extent(
                    [[0, 0], [w, h]]
                ).scaleExtent(
                    [0.1, 8]
                ).on(
                    "zoom",
                    _zoomed
                )
            )

    def _draw_graph(
        self,
        current,
        svg,
        dx=None,
        duration=750,
        _show_circles=False,
    ):

        width, height = self.parent.size

        ########################################################################
        ## Dimensions setup
        ########################################################################

        visible_depth = _max_visible_depth(self.root_tree)
        max_num_children = self.root_tree.data.num_descendants
        _node_radius_descendants = lambda num: num/max_num_children
        _node_radius = lambda d: (
            max(
                5,
                _MAX_RADIUS * _node_radius_descendants(d.data.num_descendants)
            ) if d.data.depth else _MAX_RADIUS//2
        )
        if dx:
            self.root_tree.dx = dx
        else:
            expanded_depth = _fully_expanded_depth(self.root_tree)
            self.root_tree.dx = max(
                (
                    len(self.root_tree.children) * 0.8
                    if expanded_depth > float("inf") else (
                        len(self.root_tree.children or []) * 6
                    )
                ),
                _MIN_DX
            )
        self._treemap = d3.tree().nodeSize(
            [self.root_tree.dx, self.root_tree.dy]
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

        node = svg.selectAll("g.node").attr(
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
            self._draw_graph(d, svg, dx, duration)

        # Create our nodes with our recursive clicking function
        node_enter = node.enter().append("g").attr(
            "class",
            "node",
        ).attr(
            "transform",
            lambda d: f"translate({current.y0}, {current.x0})",
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
        for i, cls_name in enumerate(self.root_tree.data.class_counts):
            _class_to_color[cls_name] = colors(i)

        node_enter.selectAll("g").data(
            _derp
        ).enter().append("path").attr(
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
                f"<b>{d.key}</b>: {d.percent*100:.3f}%"
                if d.children else f"<b>score</b>: {d.data.score}",
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

        if _show_circles:
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
            "x",
            lambda d: (
                -(_node_radius(d) + 3) if d.children or d._children
                else (_node_radius(d) + 3)
            ),
        ).attr(
            "dy",
            "0.35em"
        ).attr(
            "text-anchor",
            lambda d: "end" if d.children or d._children else "start"
        ).html(
            lambda d: d.data.name,
        ).attr(
            "font-weight",
            lambda d: "normal" if d.children or d._children else "bold",
        ).attr(
            "font-size",
            lambda d: 10 if d.children or d._children else 20,
        )

        # Transition nodes to their new position
        node_update = node_enter.merge(node)
        node_update.transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({d.y}, {d.x})",
        )

        if _show_circles:
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
        )

        # Transition function for the node exit
        # First remove our node
        node_exit = node.exit().transition().duration(
            duration
        ).attr(
            "transform",
            lambda d: f"translate({current.y}, {current.x})",
        ).remove()

        if _show_circles:
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
        link = svg.selectAll("path.link").data(
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
        link_update = link_enter.merge(link)
        link_update.transition().duration(
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

    def load_viz(self):
        self._id_count = 0
        width, height = self.parent.size
        self.root_tree = d3.hierarchy(
            self.data,
            lambda d: d.children,
        )

        self.root_tree.dx = 15
        self.root_tree.dy = 70 * _max_name_length(self.data)
        self.root_tree.x0 = height/2
        self.root_tree.y0 = 0
        self._treemap = d3.tree().nodeSize(
            [self.root_tree.dx, self.root_tree.dy]
        ).separation(
            lambda a, b: 2 if a.parent == b.parent else 3
        )

        x = d3.select('#' + self.id)
        svg = x.append("svg").attr(
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

        # Collapse all children of root
        def _collapse_all(root_too=False):
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
            else:
                if self.root_tree.children:
                    self.root_tree.children.forEach(_collapse)
            self._draw_graph(self.root_tree, svg)

        # Draw the actual graph after collapsing to the first level
        self._collapse_tree = _collapse_all
        self._collapse_tree()

        # Also set up our expander action in case we want to use this later
        # on
        def _expand_all():
            def _expand_node(d):
                if hasattr(d, "_children") and (d._children):
                    d.children = d._children
                    d._children = None
                if d.children:
                    d.children.forEach(_expand_node)
            if not self.root_tree.children:
                if (
                    hasattr(self.root_tree, "_children") and
                    self.root_tree._children
                ):
                    self.root_tree.children = self.root_tree._children
                    self.root_tree._children = None
            self.root_tree.children.forEach(_expand_node)
            self._draw_graph(
                self.root_tree,
                svg,
                len(self.root_tree.children) * 0.75,  # dx to take in account
                                                      # elongation
            )
        self._expand_tree = _expand_all


class RuleExplorerComponent(CamvizWindow):

    def _compute_hierarchical_tree(self):
        return _ruleset_hierarchy_tree(ruleset=self.root.state.ruleset)

    def init(self):
        with ui.HBox(
            title="Rule Explorer",
        ):
            with ui.VBox(style="background: #fafafa;"):
                ui.Widget(flex=1)
                self.expand_button = flx.Button(text='Expand all')
                self.collapse_button = flx.Button(text='Collapse all')
                ui.Widget(flex=1)
            with ui.VBox(flex=1):
                self.tree = HierarchicalTreeViz(
                    data=self._compute_hierarchical_tree()
                )

    @flx.reaction('expand_button.pointer_click')
    def _expand_clicked(self, *events):
        self.tree.expand_tree()

    @flx.reaction('collapse_button.pointer_click')
    def _collapse_clicked(self, *events):
        self.tree.collapse_tree()

    @flx.action
    def reset(self):
        # TODO
        pass
