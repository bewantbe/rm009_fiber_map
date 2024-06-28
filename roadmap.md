Done
----
* Re-factory
  - make map-ploting of multiple modes easier. Mostly PlotInPyvista().
* plot each classical colored topo map for V1.
* make a new type of plot: color as distanct to center,
    with an arrow to indicate the direction.
* make x-axis Eccentricity, and y the Inclination.
* Fix the color of left-right eye V1 map.
  - make the two colors even, and make the mixed area transparent.
* Plot also the neuron index in the 2D map, for fast debugging.

In progress
-----------

Under consideration
-------------------
* Refine the tree filtering.
* Try more types of coloring
  - e.g. color as the distance to the soma
  - draw ellipse to indicate the spread of the terminals
  - Try different definition of terminals.
* Use voxel based mask for LGN layer identification.
* registrate the LGN mask with LGN standard layer model.
* Use standardized LGN, V1 coordinate
* Compatitable with curved manifold for projection
