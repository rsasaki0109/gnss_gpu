"""Replace recursive BVH scheduling while copying every numerical statement."""

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "bvh_baseline_20261019"
SHA256 = "7d9d06f128710f12eea70b09b8f0d475d16223957ddfdf41a7cf5ec77ac09d98"


def project(source=None):
    raw = (BASE / "src/raytrace/bvh.cu").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    original = raw.decode().replace("\r\n", "\n")
    source = original if source is None else source.replace("\r\n", "\n")
    begin = original.index("static int build_recursive(")
    end = original.index("\nvoid bvh_build(", begin)
    old = original[begin:end]
    assert source.count(old) == 1
    opening = old.index("{")
    body = old[opening + 1 : old.rfind("}")]
    assert body.count("    return idx;") == 1
    body = body.replace("    return idx;", "    attach_parent(idx);\n    continue;")
    recursive = """  int left_child = build_recursive(geometry, indices, start, best_split, nodes,
                                   prefix_box, suffix_box);
  int right_child = build_recursive(geometry, indices, best_split, end, nodes,
                                    prefix_box, suffix_box);

"""
    assert body.count(recursive) == 1
    body = body.replace(recursive, "")
    body = body.replace(".left = left_child;", ".left = -1;")
    body = body.replace(".right = right_child;", ".right = -1;")
    body = body.replace(
        "  return node_idx;",
        """  attach_parent(node_idx);
  // Right is deferred so the left subtree gets the same node indices first.
  pending.push_back({best_split, end, node_idx, true});
  pending.push_back({start, best_split, node_idx, false});""",
    )
    assert "build_recursive(" not in body
    signature = """static void build_iterative(const BuildTriangle* geometry,
                           std::vector<int>& indices, int first, int last,
                           std::vector<BVHNode>& nodes,
                           AABB* prefix_box, AABB* suffix_box) {
  struct Work { int start, end, parent; bool right; };
  std::vector<Work> pending;
  pending.reserve(64);
  pending.push_back({first, last, -1, false});
  while (!pending.empty()) {
    const Work task = pending.back();
    pending.pop_back();
    int start = task.start;
    int end = task.end;
    auto attach_parent = [&](int index) {
      if (task.parent >= 0) {
        if (task.right) nodes[task.parent].right = index;
        else nodes[task.parent].left = index;
      }
    };
"""
    new = (
        signature
        + "\n".join("  " + s if s else s for s in body.split("\n"))
        + "\n  }\n}\n"
    )
    output = source.replace(old, new)
    old_call = "  build_recursive(geometry.data(), indices, 0, n_tri, node_vec,"
    new_call = "  build_iterative(geometry.data(), indices, 0, n_tri, node_vec,"
    assert output.count(old_call) == 1
    output = output.replace(old_call, new_call)
    restored = output.replace(new_call, old_call).replace(new, old)
    assert restored == source
    # All source after the GPU boundary stays byte-for-byte unchanged.
    boundary = "// GPU-side BVH traversal"
    assert output[output.index(boundary) :] == source[source.index(boundary) :]
    return output, dict(
        full_source_undo_exact=True,
        all_GPU_source_unchanged=True,
        original_baseline_sha256=SHA256,
        input_sha256=hashlib.sha256(source.replace("\n", "\r\n").encode()).hexdigest(),
        output_sha256=hashlib.sha256(output.replace("\n", "\r\n").encode()).hexdigest(),
        numerical_prefix_copied=True,
        split_tie_sort_and_node_order_unchanged=True,
    )


def binding_marker(source):
    mark = "PYBIND11_MODULE(_bvh, m) {"
    assert source.count(mark) == 1 and 'm.attr("_bvh_iterative_build")' not in source
    new = mark + '\n  m.attr("_bvh_iterative_build") = true;'
    output = source.replace(mark, new)
    assert output.replace(new, mark) == source
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    output, proof = project()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output.replace("\n", "\r\n").encode())
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
