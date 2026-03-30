import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------
# 1. Load data and keep only the two requested features
# ------------------------------------------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

feature_names = ["Longitude", "Latitude"]
target_name = "MedHouseVal"

if len(feature_names) != 2:
    raise ValueError("feature_names must contain exactly two feature names.")

X = df[feature_names]
y = df[target_name]

x_feature, y_feature = feature_names

# ------------------------------------------------------------
# 2. Fit a depth-2 decision tree regressor
# ------------------------------------------------------------
tree = DecisionTreeRegressor(max_depth=2, random_state=42)
tree.fit(X, y)

# ------------------------------------------------------------
# 3. Extract split structure from the fitted tree
# ------------------------------------------------------------
tree_ = tree.tree_
children_left = tree_.children_left
children_right = tree_.children_right
feature = tree_.feature
threshold = tree_.threshold
value = tree_.value.squeeze()


def is_leaf(node_id):
    return children_left[node_id] == children_right[node_id]


def node_prediction(node_id):
    return float(value[node_id])


def node_label(node_id):
    if is_leaf(node_id):
        return f"Leaf:\ny_hat = {node_prediction(node_id):.2f}"
    feat_name = feature_names[feature[node_id]]
    thr = threshold[node_id]
    return f"{feat_name} <= {thr:.2f}?"


root = 0
left_1 = children_left[root]
right_1 = children_right[root]

left_2_left = children_left[left_1] if not is_leaf(left_1) else None
left_2_right = children_right[left_1] if not is_leaf(left_1) else None
right_2_left = children_left[right_1] if not is_leaf(right_1) else None
right_2_right = children_right[right_1] if not is_leaf(right_1) else None

# ------------------------------------------------------------
# 4. Compute leaf rectangles in feature space
# ------------------------------------------------------------
x_min = X[x_feature].min()
x_max = X[x_feature].max()
y_min = X[y_feature].min()
y_max = X[y_feature].max()

rectangles = []


def collect_regions(node_id, xmin, xmax, ymin, ymax):
    if is_leaf(node_id):
        rectangles.append(
            {
                "node": node_id,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "pred": node_prediction(node_id),
            }
        )
        return

    feat_idx = feature[node_id]
    thr = threshold[node_id]
    feat_name = feature_names[feat_idx]

    if feat_name == x_feature:
        collect_regions(children_left[node_id], xmin, min(xmax, thr), ymin, ymax)
        collect_regions(children_right[node_id], max(xmin, thr), xmax, ymin, ymax)
    elif feat_name == y_feature:
        collect_regions(children_left[node_id], xmin, xmax, ymin, min(ymax, thr))
        collect_regions(children_right[node_id], xmin, xmax, max(ymin, thr), ymax)
    else:
        raise ValueError(f"Unexpected feature: {feat_name}")


collect_regions(root, x_min, x_max, y_min, y_max)

# ------------------------------------------------------------
# 5. Fixed layout for the depth-2 tree drawing
# ------------------------------------------------------------
positions = {
    root: (0.50, 0.88),
    left_1: (0.25, 0.60),
    right_1: (0.75, 0.60),
}

if left_2_left is not None:
    positions[left_2_left] = (0.12, 0.30)
if left_2_right is not None:
    positions[left_2_right] = (0.38, 0.30)
if right_2_left is not None:
    positions[right_2_left] = (0.62, 0.30)
if right_2_right is not None:
    positions[right_2_right] = (0.88, 0.30)


def draw_edge(ax, parent, child, text):
    x1, y1 = positions[parent]
    x2, y2 = positions[child]
    ax.add_line(Line2D([x1, x2], [y1 - 0.04, y2 + 0.05], linewidth=2))
    x_offset = -0.03 if x2 < x1 else 0.03
    ax.text(
        (x1 + x2) / 2 + x_offset,
        (y1 + y2) / 2 + 0.02,
        text,
        fontsize=10,
        ha="center",
        va="center",
    )


def draw_node(ax, node_id):
    x, y = positions[node_id]
    text = node_label(node_id)
    leaf = is_leaf(node_id)

    width = 0.26 if not leaf else 0.22
    height = 0.11 if not leaf else 0.10

    rect = Rectangle(
        (x - width / 2, y - height / 2), width, height, fill=False, linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=11, ha="center", va="center")


def draw_tree(ax):
    ax.axis("off")
    ax.set_title("DecisionTreeRegressor (max_depth=2)", fontsize=15, pad=15)

    draw_edge(ax, root, left_1, "yes")
    draw_edge(ax, root, right_1, "no")

    if not is_leaf(left_1):
        draw_edge(ax, left_1, left_2_left, "yes")
        draw_edge(ax, left_1, left_2_right, "no")

    if not is_leaf(right_1):
        draw_edge(ax, right_1, right_2_left, "yes")
        draw_edge(ax, right_1, right_2_right, "no")

    for node_id in positions:
        draw_node(ax, node_id)


def draw_split_lines(ax, node_id, xmin, xmax, ymin, ymax):
    if is_leaf(node_id):
        return

    feat_idx = feature[node_id]
    thr = threshold[node_id]
    feat_name = feature_names[feat_idx]

    if feat_name == x_feature:
        ax.plot([thr, thr], [ymin, ymax], linewidth=2, color="black")
        ax.text(
            thr + 0.01 * (x_max - x_min),
            ymax,
            f"{x_feature} = {thr:.2f}",
            rotation=90,
            va="top",
            fontsize=10,
        )
        draw_split_lines(ax, children_left[node_id], xmin, thr, ymin, ymax)
        draw_split_lines(ax, children_right[node_id], thr, xmax, ymin, ymax)

    elif feat_name == y_feature:
        ax.plot([xmin, xmax], [thr, thr], linewidth=2, color="black")
        ax.text(
            xmin + 0.01 * (x_max - x_min),
            thr + 0.01 * (y_max - y_min),
            f"{y_feature} = {thr:.2f}",
            va="bottom",
            fontsize=10,
        )
        draw_split_lines(ax, children_left[node_id], xmin, xmax, ymin, thr)
        draw_split_lines(ax, children_right[node_id], xmin, xmax, thr, ymax)


def draw_feature_space(ax, show_points):
    ax.set_title("Same Splits in Feature Space", fontsize=15, pad=10)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for region in rectangles:
        rect = Rectangle(
            (region["xmin"], region["ymin"]),
            region["xmax"] - region["xmin"],
            region["ymax"] - region["ymin"],
            fill=True,
            alpha=0.12,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        cx = (region["xmin"] + region["xmax"]) / 2
        cy = (region["ymin"] + region["ymax"]) / 2
        ax.text(
            cx,
            cy,
            f"y_hat = {region['pred']:.2f}",
            ha="center",
            va="center",
            fontsize=11,
        )

    draw_split_lines(ax, root, x_min, x_max, y_min, y_max)

    if show_points:
        ax.scatter(
            X[x_feature],
            X[y_feature],
            s=3,
            alpha=0.03,
            color="black",
        )


# ------------------------------------------------------------
# 6. Save three separate PNGs
# ------------------------------------------------------------
fig_tree, ax_tree = plt.subplots(figsize=(8, 6))
draw_tree(ax_tree)
fig_tree.tight_layout()
fig_tree.savefig("tree_regressor_tree.png", dpi=200, bbox_inches="tight")
plt.close(fig_tree)

fig_box, ax_box = plt.subplots(figsize=(8, 6))
draw_feature_space(ax_box, show_points=False)
fig_box.tight_layout()
fig_box.savefig("tree_regressor_boxes.png", dpi=200, bbox_inches="tight")
plt.close(fig_box)

fig_box_points, ax_box_points = plt.subplots(figsize=(8, 6))
draw_feature_space(ax_box_points, show_points=True)
fig_box_points.tight_layout()
fig_box_points.savefig("tree_regressor_boxes_points.png", dpi=200, bbox_inches="tight")
plt.close(fig_box_points)

print("Saved:")
print("  tree_regressor_tree.png")
print("  tree_regressor_boxes.png")
print("  tree_regressor_boxes_points.png")
