# Woodland rendering

Open `forest.html` for the receiver-height trail view. Drag to look around,
scroll to change the field of view, and use **Overview** for the scene map.
The **Light** control switches between filtered sunlight and diffuse overcast
illumination. Tree type, season, seed, route time and signal inspection remain
available in both views.

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
node --test tests/site/forest-appearance.test.cjs tests/site/forest-model.test.cjs tests/site/forest-species.test.cjs
npm run site:smoke -- tests/site/forest.spec.cjs --grep 'woodland lighting|trail perspective|tree type and season'
```
