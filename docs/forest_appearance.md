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
node --test tests/site/forest-live-history.test.cjs tests/site/forest-signal-overlay.test.cjs tests/site/forest-appearance.test.cjs tests/site/forest-model.test.cjs tests/site/forest-species.test.cjs
npm run site:smoke -- tests/site/forest.spec.cjs --grep 'live signal charts|GNSS paths|woodland lighting|trail perspective|tree type and season'
```
