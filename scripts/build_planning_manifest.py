#!/usr/bin/env python3
"""Build route-split planning QA manifests with resolved graph context."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def references(value):
    if value in (None, -1):
        return []
    return [tuple(ref) for ref in value]


def linked_references(node):
    """Return the unique graph neighbors in the official link order.

    DriveLM stores links in two directions.  Some planning nodes, such as
    braking, have no ``con_up`` links but do have supporting ``con_down``
    links, so using only one direction silently drops graph evidence.
    """
    refs = []
    seen = set()
    for ref in node["con_up"] + node["con_down"]:
        if ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def resolve_image(image_ref: str, image_root: Path) -> Path:
    parts = Path(image_ref).parts
    if "data" in parts:
        relative = Path(*parts[parts.index("data") + 1 :])
    else:
        relative = Path(image_ref)
    return image_root / relative


def load_nodes(frame_data):
    rows = []
    by_cluster = defaultdict(list)

    for task, items in frame_data.get("QA", {}).items():
        for item in items:
            node = {
                "task": task,
                "question": item["Q"],
                "answer": item["A"],
                "cluster": item.get("cluster"),
                "layer": item.get("layer"),
                "con_up": references(item.get("con_up")),
                "con_down": references(item.get("con_down")),
            }
            by_cluster[node["cluster"]].append(node)
            rows.append(node)

    for cluster, nodes in by_cluster.items():
        for local_index, node in enumerate(nodes):
            node["node_id"] = f"{cluster}:{local_index}"

    return rows, by_cluster


def resolve_context(refs, by_cluster):
    context = []
    for cluster, local_index in refs:
        nodes = by_cluster.get(cluster, [])
        if 0 <= local_index < len(nodes):
            node = nodes[local_index]
            context.append(
                {
                    "node_id": node["node_id"],
                    "task": node["task"],
                    "question": node["question"],
                    "answer": node["answer"],
                    "cluster": node["cluster"],
                    "layer": node["layer"],
                }
            )
    return context


def resolve_linked_context(node, by_cluster):
    """Resolve both graph directions and retain how each link was reached."""
    directions = defaultdict(list)
    for direction, refs in (
        ("up", node["con_up"]),
        ("down", node["con_down"]),
    ):
        for ref in refs:
            directions[ref].append(direction)

    context = []
    for ref in linked_references(node):
        resolved = resolve_context([ref], by_cluster)
        if not resolved:
            continue
        item = resolved[0]
        item["link_directions"] = directions[ref]
        context.append(item)
    return context


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text())
    group_to_split = {
        group: "train" for group in split["train_groups"]
    }
    group_to_split.update(
        {group: "validation" for group in split["validation_groups"]}
    )

    manifests = {"train": [], "validation": []}
    frame_count = 0
    missing_images = []

    for route_dir in sorted(path for path in args.qa_root.iterdir() if path.is_dir()):
        route_group = route_dir.name.split("_Rep", 1)[0]
        split_name = group_to_split.get(route_group)
        if split_name is None:
            raise ValueError(f"Route group missing from split: {route_group}")

        for frame_path in sorted(route_dir.glob("*.json")):
            frame_data = json.loads(frame_path.read_text())
            image_ref = frame_data.get("image_paths", {}).get("CAM_FRONT")
            if not image_ref:
                missing_images.append(str(frame_path))
                continue

            image_path = resolve_image(image_ref, args.image_root)
            if not image_path.exists():
                missing_images.append(str(image_path))
                continue

            nodes, by_cluster = load_nodes(frame_data)
            frame_count += 1

            for node in nodes:
                if node["task"] != "planning":
                    continue

                row = {
                    "frame_json": str(frame_path),
                    "route_group": route_group,
                    "route_rep": route_dir.name,
                    "split": split_name,
                    "image_path": str(image_path),
                    "node_id": node["node_id"],
                    "cluster": node["cluster"],
                    "layer": node["layer"],
                    "question": node["question"],
                    "answer": node["answer"],
                    "con_up": node["con_up"],
                    "con_down": node["con_down"],
                    # For a planning target, DriveLM's graph prompt uses the
                    # preceding/parent QA nodes.  In the CARLA serialization,
                    # these are the node references stored in con_down.
                    "parent_context": resolve_context(node["con_down"], by_cluster),
                    # Keep the bidirectional neighborhood for diagnostics,
                    # but do not use it as the training context.
                    "linked_context": resolve_linked_context(node, by_cluster),
                    # Retained for compatibility with the first pilot
                    # manifests.  New experiments should use linked_context.
                    "upstream_context": resolve_context(node["con_up"], by_cluster),
                }
                manifests[split_name].append(row)

    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        for path in missing_images[:10]:
            print(" ", path)
        raise SystemExit("Manifest creation stopped because images are missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in manifests.items():
        output = args.output_dir / f"planning_{split_name}.jsonl"
        with output.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"{split_name}: {len(rows)} planning examples -> {output}")

    print(f"Frames processed: {frame_count}")


if __name__ == "__main__":
    main()
