// D3.js Visualization for PCA space
let svg, g, zoom;
let pointsData = [];
let connectionsData = [];

// Initialize visualization
function initializeVisualization(points, connections) {
    pointsData = points;
    connectionsData = connections;

    // Clear existing SVG
    d3.select('#visualization').selectAll('*').remove();

    // Get container dimensions
    const container = document.getElementById('visualization');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Create SVG
    svg = d3.select('#visualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create zoom behavior
    zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Disable double click zoom
    svg.on("dblclick.zoom", null);

    // Create main group for zoom/pan
    g = svg.append('g');

    // Calculate data bounds and scales
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;

    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([width * 0.1, width * 0.9]); // Keep away from edges

    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height * 0.9, height * 0.1]); // D3 y is top-down

    // Transform points to screen coordinates
    points.forEach(p => {
        p.screenX = xScale(p.x);
        p.screenY = yScale(p.y);
    });

    // LAYERS ORDER: Connections (bottom) -> Polygons -> Points (top)
    g.append('g').attr('class', 'connections-layer');
    g.append('g').attr('class', 'polygons-layer');
    g.append('g').attr('class', 'points-layer');

    renderConnections();
    renderPolygons();
    renderPoints();

    setupInteraction();
    resetVisualizationZoom();
}

function renderConnections() {
    const layer = g.select('.connections-layer');
    layer.selectAll('*').remove();

    layer.selectAll('.connection')
        .data(connectionsData)
        .enter()
        .append('line')
        .attr('class', 'connection')
        .attr('x1', d => pointsData[d.source].screenX)
        .attr('y1', d => pointsData[d.source].screenY)
        .attr('x2', d => pointsData[d.target].screenX)
        .attr('y2', d => pointsData[d.target].screenY);
}

function renderPoints() {
    const layer = g.select('.points-layer');
    layer.selectAll('*').remove();

    // Create points
    const circles = layer.selectAll('.point')
        .data(pointsData)
        .enter()
        .append('circle')
        .attr('class', 'point')
        .attr('cx', d => d.screenX)
        .attr('cy', d => d.screenY)
        .attr('r', 5)
        .attr('fill', d => {
            // Color check
            const poly = window.polygonManager.getPolygonForPoint(d.id);
            return poly ? poly.color : '#fff'; // Default white
        })
        .classed('selected', d => window.appState.selectedPoint && window.appState.selectedPoint.id === d.id);

    // Add interaction
    circles.on('mouseover', function () {
        d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 8);
    })
        .on('mouseout', function () {
            // Only revert if NOT selected
            const isSelected = d3.select(this).classed('selected');
            d3.select(this)
                .transition()
                .duration(200)
                .attr('r', isSelected ? 8 : 5);
        })
        .on('click', (event, d) => {
            // Stop propagation so it doesn't trigger map click
            event.stopPropagation();

            // Select point logic
            if (window.handlePointSelection) {
                d3.selectAll('.point').classed('selected', false).transition().attr('r', 5);
                d3.select(event.currentTarget).classed('selected', true).transition().attr('r', 8);
                window.handlePointSelection(d);
            }
        });
}


function renderPolygons() {
    const layer = g.select('.polygons-layer');
    layer.selectAll('*').remove();

    if (!window.polygonManager) return;

    const polygons = window.polygonManager.polygons;
    const current = window.polygonManager.currentPolygon;
    const selected = window.polygonManager.selectedPolygon;
    const mode = window.polygonManager.mode;

    // Render saved polygons
    polygons.forEach(poly => {
        drawPolygonShape(layer, poly, false, poly === selected, mode);
    });

    // Render current being drawn
    if (current && current.vertices.length > 0) {
        drawPolygonShape(layer, current, true, false, mode);
    }
}

function drawPolygonShape(layer, poly, isDrawing, isSelected, mode) {
    const group = layer.append('g').attr('class', 'polygon-group');

    const vertices = poly.vertices;
    if (vertices.length === 0) return;

    let pathD = `M ${vertices[0].x} ${vertices[0].y}`;
    for (let i = 1; i < vertices.length; i++) {
        pathD += ` L ${vertices[i].x} ${vertices[i].y}`;
    }
    if (!isDrawing) pathD += ' Z'; // Close path if finished

    // Shape
    const path = group.append('path')
        .attr('class', 'polygon-shape')
        .attr('d', pathD)
        .attr('stroke', poly.color)
        .attr('fill', poly.color)
        .classed('selected', isSelected);

    // Label
    if (!isDrawing && vertices.length > 0) {
        const cx = d3.mean(vertices, d => d.x);
        const cy = d3.mean(vertices, d => d.y);
        group.append('text')
            .attr('class', 'polygon-label')
            .attr('x', cx)
            .attr('y', cy)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .text(poly.name);
    }

    // Interaction on Polygon
    if (!isDrawing) {
        path.on('click', (event) => {
            if (mode === 'select' || mode === 'move' || mode === 'edit') {
                event.stopPropagation();
                window.polygonManager.selectPolygonAt(d3.pointer(event, g.node())[0], d3.pointer(event, g.node())[1]);
                renderPolygons(); // Re-render to show selection
                renderPoints();
            }
        });

        // Move drag handler
        if (mode === 'move' && isSelected) {
            path.call(d3.drag()
                .on('start', (event) => {
                    const [x, y] = d3.pointer(event, g.node());
                    window.polygonManager.startMove(x, y);
                })
                .on('drag', (event) => {
                    const [x, y] = d3.pointer(event, g.node());
                    window.polygonManager.move(x, y);
                    renderPolygons();
                    renderPoints(); // Update points coloring in real time
                })
                .on('end', () => {
                    window.polygonManager.endMove();
                })
            );
            // Cursor hint
            path.style('cursor', 'move');
        }
    }

    // Edit Handles (Vertices)
    if (mode === 'edit' && isSelected) {
        vertices.forEach((v, i) => {
            group.append('circle')
                .attr('class', 'polygon-vertex')
                .attr('cx', v.x)
                .attr('cy', v.y)
                .attr('r', 6)
                .call(d3.drag()
                    .on('drag', (event) => {
                        const [x, y] = d3.pointer(event, g.node());
                        if (window.polygonManager.moveVertex(i, x, y)) {
                            renderPolygons();
                            renderPoints();
                        }
                    })
                );
        });
    }
}

function setupInteraction() {
    svg.on('click', (event) => {
        // If clicking on empty space
        if (event.defaultPrevented) return; // Dragged or stopped propagation

        const [x, y] = d3.pointer(event, g.node());
        const mode = window.polygonManager.mode;

        if (mode === 'draw') {
            window.polygonManager.addVertex(x, y);
            renderPolygons();
        } else if (mode === 'select') {
            // Click outside -> deselect? Or try select at point (handled by path click usually, 
            // but if hole in polygon? Raycasting handles it)
            const poly = window.polygonManager.selectPolygonAt(x, y);
            renderPolygons();
        }
    });

    // Double click to close
    svg.on('dblclick', (event) => {
        if (window.polygonManager.mode === 'draw') {
            event.preventDefault();
            window.polygonManager.closePolygon();
        }
    });

    // Right click to close
    svg.on('contextmenu', (event) => {
        if (window.polygonManager.mode === 'draw') {
            event.preventDefault();
            window.polygonManager.closePolygon();
        }
    });
}

function resetVisualizationZoom() {
    if (!svg || !g) return;
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}

// Highlights connections for a specific point
function highlightConnections(pointId) {
    if (!g) return;

    // Reset all first
    g.selectAll('.connection')
        .classed('highlighted', false)
        .classed('dimmed', true); // Dim all initially

    // Highlight specific
    g.selectAll('.connection')
        .filter(d => d.source === pointId || d.target === pointId)
        .classed('highlighted', true)
        .classed('dimmed', false)
        .raise(); // Bring to front
}

// Resets connection highlights
function resetHighlights() {
    if (!g) return;
    g.selectAll('.connection')
        .classed('highlighted', false)
        .classed('dimmed', false);
}

// Select a point programmatically (and zoom to it optional)
function selectPointById(pointId) {
    const point = pointsData.find(p => p.id === pointId);
    if (point) {
        // Trigger visual selection
        d3.selectAll('.point').classed('selected', false).transition().attr('r', 5);

        // Find specific circle (filtering by data)
        d3.selectAll('.point').filter(d => d.id === pointId)
            .classed('selected', true)
            .transition().attr('r', 8);

        // Trigger app logic
        if (window.handlePointSelection) {
            window.handlePointSelection(point);
        }
    }
}


// Export functions
window.initializeVisualization = initializeVisualization;
window.renderPolygons = renderPolygons;
window.renderPoints = renderPoints;
window.resetVisualizationZoom = resetVisualizationZoom;
window.highlightConnections = highlightConnections;
window.resetHighlights = resetHighlights;
window.selectPointById = selectPointById;
