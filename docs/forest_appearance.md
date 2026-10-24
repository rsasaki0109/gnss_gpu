# Woodland rendering

Open `forest.html` for the receiver-follow trail view. Drag to look around,
scroll to change the field of view, and use **Overview** for the scene map.
The **Light** control switches between filtered sunlight and diffuse overcast
illumination. Tree type, season, seed, route time and signal inspection remain
available in both views.

GNSS paths are enabled for all satellites by default. The observer follows
6 m behind the antenna horizontally and 1 m above it; placing the camera at
the antenna itself would collapse incoming rays into single screen points.
The physical receiver route and its 1.6 m terrain offset remain unchanged.
**Signal paths** switches between all satellites, the selected satellite and
hidden paths. Click a direction label to select a signal; **Satellite** points
the camera toward its arrival direction. **Overview** restores the scene map.

The screen-space overlay uses a shared camera projection and clips segments
at the near plane and viewport. It draws coloured paths with arrival arrows,
satellite identifiers, the selected canopy-loss value, selected crown/wood
intersections, the RX marker and both legs of the selected valid ground bounce.
Blocked direct geometric alternatives are dashed. These paths deliberately
overlay occluding scenery; they are not an assertion that blocked signals reach
the antenna. Ray extents are 160 m along the computed arrival directions;
edge labels indicate direction, not spacecraft positions at those distances.
Arrows show propagation direction, not RF phase or real-time wave speed.

## Changes during playback

The selected satellite now has live charts below the viewer: direct canopy
loss in dB and the shortest valid ground path's excess range in metres.
Current values, change in loss since the preceding sample, geometry-class
transitions, ground path gain/loss and ground triangle changes accompany them.
The chart fills the available recorded interval, up to 120 simulated seconds;
its cursor follows the latest completed epoch. No extra route scan or future
orbit/propagation calculation is performed for these plots.

History retains at most 600 completed samples. Rendering again at the same
timestamp replaces its sample rather than duplicating it. Satellite selection,
RF/forest settings changes, timeline seeks, backward time or a gap over five
simulated seconds start a new history segment. Camera, lighting and path-display
changes preserve the recorded measurements. Blocked direct paths and absent
ground paths have null values: the curves stop rather than connecting across
outages or assigning a zero loss/excess range. Missing a sampled reflection
does not establish its exact loss/recovery time between completed epochs.

The latest 30 ground reflection points form a dotted amber trail in the 3D
overlay. Missing reflection epochs break the trail; it is historical geometry,
not additional simultaneous multipath. Hiding signal paths hides this trail.
The snapshot JSON includes an immutable `signalHistory` record with schema
`forest-live-history-v1`, sample times relative to the existing scene start,
model values, geometry states, selected ground triangles/points and events.
These remain illustrative model values, not C/N0, measured receiver strength,
coherently combined signals or a receiver-tracking result.

## Which trees contribute to the loss?

The **Loss by tree / 原因の木** panel decomposes the selected satellite's
computed crown loss by tree ID. Choose the direct ray or the shortest valid
ground-reflected path. Ground attribution sums both complete traced legs;
fixed bounce loss and geometric spreading remain separate. No extra rays or
alternative RF model are evaluated for this panel.

In linear mode, each crown component contributes specific loss × foliage
multiplier × its traced density integral. In the extinction reference, it
contributes the same integral × that crown's reference dB/m coefficient.
Components add by tree, including overlapping densities. The sum is checked
against the completed trace's canopy loss; mismatched settings cannot silently
produce an apparently reconciled attribution. Per-tree crown path is the union
of that tree's intervals on each leg, summed over distinct traversals. Path
lengths across different overlapping trees are not additive union lengths.

Wood intersections are listed separately, with no invented wood dB loss.
For blocked direct paths the canopy subtotal is a geometric component, not
total received power. A missing ground path stays unavailable rather than
reporting zero attenuation. The table orders additive crown dB contributions,
not percentages of signal energy or calibrated causal effects on real data.

Clicking a tree pauses playback, focuses an overview on it and outlines its
model crown ellipsoid(s) in blue. These are the same shape parameters used by
the ray integrator, not extra scattering or attenuation volumes. Inspection
switches the path display to the selected satellite; hiding paths also hides
the outline. Satellite/route/forest changes update the table and its current
tree selection. Snapshot JSON retains `treeAttribution` with satellite, scope,
per-tree loss/length/wood intersections and the reconciliation totals. Inspecting
a tree changes the view, not the completed signal measurements or history.
Tree inspection buttons remain attached while completed epochs update, so a
slow rendering frame cannot detach the input target before its click completes.

## Paired leaf-on / leaf-off comparison

Open **Leaf-on vs leaf-off / 着葉・落葉を同じ条件で比較** and select
**Compare current epoch**. Playback pauses and captures the current receiver,
visible satellite endpoints, selected signal, forest geometry, RF assumptions
and completed epoch. Both cases retain the same tree placement, wood, crown
component geometry and density; only the existing binary leaf-retention rule
changes. The live scene, selected season and recorded measurements are not
replaced by the comparison cases.

Every captured satellite gets leaf-on/off crown-loss components and their
off-minus-on difference, ordered by the magnitude of that difference. Wood
and terrain blockage remain visible; the crown values are not total received
power for blocked rays. The selected satellite also gets both full ground
queries, showing the shortest valid path's scalar relative power and geometric
excess range. Bounce loss is fixed. No noise, tracking, weather, forest growth,
coherent combination or positioning model is added by this comparison.

In linear mode, broadleaf leaf-off removes crown leaf-density loss by
assumption. In the extinction reference it retains the existing bare-crown
coefficient (maple ratio 0.139/0.241). Evergreen foliage and all wood remain in
both seasons. Consequently this is a controlled model sensitivity comparison,
not a field estimate of seasonal GNSS performance or calibrated received power.

Results remain the captured study while the live scene changes. A warning
appears when receiver time/position, satellite set/selection, forest layout/type
or RF assumptions no longer match. Display season alone does not invalidate
the study because it already contains both seasons; camera and lighting changes
also preserve it. Run again to capture new conditions.

**Export comparison** writes a separate `forest-season-comparison-v1` JSON
containing the complete captured forest, endpoints and context, both direct
trace sets, selected ground queries, scalar differences and interpretation
assumptions. Missing ground paths remain null/unavailable. This independent
export avoids mixing an older comparison with the current live snapshot.

The display now uses botanical branch cards rather than solid ellipsoid
shells. A locally generated four-region texture atlas supplies broad leaves,
leaf veins, twigs and evergreen needles. Random orientations and layered
placement break up crown outlines and allow glimpses through the foliage.
Broadleaf leaf-off removes those crown cards; evergreen foliage remains.

Additional detail includes tapered secondary branches, root flares, fallen
wood with exposed ends, stones, curled litter, mottled moss and low understory.
The ground shader adds small leaf shapes with central veins and rough soil
variation. Bark has longitudinal ridges and cracks. Two-sided leaf lighting
and a small backlit contribution distinguish leaves from opaque wood.

A cached 1536×1536 sunlight depth map uses the same leaf alpha cutouts as the
main view. Nine filtered depth comparisons soften the shadow edges. Terrain,
wood and individual branch silhouettes therefore cast shadows onto the rest
of the scene. A separate needle coverage threshold preserves evergreen
silhouettes under texture minification. Receiver-view distance haze and a subdued atmospheric horizon
provide depth. The overcast preset increases diffuse illumination and reduces
the direct-light contribution. Wind deformation is subtle, display-only and
evaluated when the view redraws; reduced-motion preferences disable it.

All assets are generated locally from fixed seeds; there are no external
texture services or new rendering dependencies. Default 128-tree broadleaf,
conifer and mixed scenes stay below 800,000 vertices and 44 MiB of vertex data
in the tested seed. Shadow maps are rebuilt after scene or foliage changes,
not on each camera move. Canvas pixel ratio is capped at 1.5 to limit fragment
work on mobile screens.

This remains a procedural visualization, not a photographic reconstruction
or measured forest. Decorative twigs, roots, low plants, stones, litter,
fallen wood, leaf-card silhouettes and the distant horizon are not additional
RF geometry. Direct/reflected GNSS rays still use the exported binary64 terrain,
wood cylinders and crown-density volumes. Lighting and wind do not change
those calculations. Exports name these display assumptions and retain the
chosen lighting preset in `view.lighting`.

Validation covers deterministic buffers, finite unit normals/atlas coordinates,
seasonal behavior, memory bounds and unchanged RF scenes/traces. Browser checks
exercise both lighting presets, default trail/overview switching, desktop and
mobile layout, species/season controls, WebGL errors and exported signal results.

```bash
node --test tests/site/forest-season-study.test.cjs tests/site/forest-tree-causes.test.cjs tests/site/forest-live-history.test.cjs tests/site/forest-signal-overlay.test.cjs tests/site/forest-appearance.test.cjs tests/site/forest-model.test.cjs tests/site/forest-species.test.cjs
npm run site:smoke -- tests/site/forest.spec.cjs --grep 'paired seasons|tree loss attribution|live signal charts|GNSS paths|woodland lighting|trail perspective|tree type and season'
```
