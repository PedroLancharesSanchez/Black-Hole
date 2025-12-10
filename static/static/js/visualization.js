// D3.js + Canvas Visualization for PCA space
let svg, g, zoom;
let canvas, ctx;
let pointsData = [];
let connectionsData = [];
let transform = d3.zoomIdentity;
let width, height;

// Interaction state
let hoveredPoint = null;

// Initialize visualization
function initializeVisualization(points, connections) {
    pointsData = points;
    connectionsData = connections;

    const container = document.getElementById('visualization');
    container.innerHTML = ''; // Clear container

    width = container.clientWidth;
    height = container.clientHeight;

    // 1. Create Canvas (Bottom Layer)
    canvas = d3.select('#visualization')
        .append('canvas')
        .attr('width', width)
        .attr('height', height)
        .node();

    ctx = canvas.getContext('2d', { alpha: true });

    // 2. Create SVG (Top Layer for interaction & polygons)
    svg = d3.select('#visualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create main group for polygons
    g = svg.append('g').attr('class', 'svg-content');

    // Layers in SVG
    g.append('g').attr('class', 'polygons-layer');

    // 3. Scale calculation
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;

    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([width * 0.1, width * 0.9]);

    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height * 0.9, height * 0.1]);

    // Pre-calculate screen coordinates
    points.forEach(p => {
        p.sx = xScale(p.x);
        p.sy = yScale(p.y);
    });

    // 4. Setup Zoom
    zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            transform = event.transform;

            // Transform SVG Polygons
            g.attr('transform', transform);

            // Redraw Canvas
            requestAnimationFrame(renderCanvas);
        });

    svg.call(zoom)
        .on("dblclick.zoom", null);

    // 5. Initial Render
    renderCanvas();
    renderPolygons();

    // 6. Setup Interaction events on SVG
    setupInteraction();
}

function renderCanvas() {
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, width, height);

    // Apply Zoom Transform
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const pointRadius = 5 / transform.k; // Constant visual size? Or scale? Let's scale slightly but keep limit
    // Actually standard behavior is points grow with zoom or stay same. 
    // Usually scatterplots stay same size or grow slightly. 
    // Let's keep fixed screen size (inverse scale)
    const radius = 4 / transform.k;
    const selectedRadius = 7 / transform.k;
    const lineWidth = 1 / transform.k;

    // A. Draw Connections
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    connectionsData.forEach(c => {
        const source = pointsData[c.source];
        const target = pointsData[c.target];
        // Check bounds optimization could go here
        ctx.moveTo(source.sx, source.sy);
        ctx.lineTo(target.sx, target.sy);
    });
    ctx.stroke();

    // B. Draw Highlighted Connections
    // (If we implemented highlighting specific lines, draw them here on top)

    // C. Draw Points
    // Optimization: Draw by color groups?
    // For now, simple loop is fast enough for <10k points on canvas

    pointsData.forEach(p => {
        const isSelected = window.appState.selectedPoint && window.appState.selectedPoint.id === p.id;
        const isHovered = hoveredPoint && hoveredPoint.id === p.id;

        ctx.beginPath();
        let r = radius;
        if (isSelected) r = selectedRadius;
        if (isHovered) r = selectedRadius * 1.1;

        ctx.arc(p.sx, p.sy, r, 0, 2 * Math.PI);

        // Color
        let color = '#fff';
        const poly = window.polygonManager.getPolygonForPoint(p.id);
        if (poly) color = poly.color;

        if (isSelected) {
            ctx.fillStyle = '#fff';
            ctx.strokeStyle = '#7000ff';
            ctx.lineWidth = 2 / transform.k;
            ctx.fill();
            ctx.stroke();
        } else if (isHovered) {
            ctx.fillStyle = '#fff'; // Highlight white
            ctx.fill();
        } else {
            ctx.fillStyle = color;
            ctx.fill();
        }
    });

    ctx.restore();
}

function renderPoints() {
    // Wrapper to trigger canvas redraw from other modules
    renderCanvas();
}

// --- Polygons (SVG) logic remains similar ---
function renderPolygons() {
    const layer = g.select('.polygons-layer');
    layer.selectAll('*').remove();

    if (!window.polygonManager) return;
    const polygons = window.polygonManager.polygons;
    const current = window.polygonManager.currentPolygon;
    const selected = window.polygonManager.selectedPolygon;
    const mode = window.polygonManager.mode;

    polygons.forEach(poly => {
        drawPolygonShape(layer, poly, false, poly === selected, mode);
    });

    if (current && current.vertices.length > 0) {
        drawPolygonShape(layer, current, true, false, mode);
    }
}

// Re-using the drawing logic but adapting coordinate system?
// The logic uses screenX/Y stored in vertices? 
// Wait, polygon vertices were stored as screen coords in previous logic?
// In previous logic: `d3.pointer` returned screen coords relative to G if zoom was applied?
// Actually `d3.pointer` on `g.node()` returns transformed coordinates.
// So vertices are in Data Space (scaled x/y), not raw pixels?
// In previous code `xScale` mapped data to screen range (pixels).
// And zoom transformed the `g`.
// So the vertices stored in PolygonManager are in "Initial Screen Space" (0..width, 0..height).
// And since `g` is transformed, they move correctly.
// Yes. We need to maintain this consistency.

function drawPolygonShape(layer, poly, isDrawing, isSelected, mode) {
    const group = layer.append('g').attr('class', 'polygon-group');
    const vertices = poly.vertices;
    if (vertices.length === 0) return;

    let pathD = `M ${vertices[0].x} ${vertices[0].y}`;
    for (let i = 1; i < vertices.length; i++) {
        pathD += ` L ${vertices[i].x} ${vertices[i].y}`;
    }
    if (!isDrawing) pathD += ' Z';

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
            .text(poly.name)
            .style('font-size', (12 / transform.k) + 'px'); // Counter-scale text?
    }

    // Interaction handled by SVG elements naturally
    // But we need to ensure events don't bubble if we want to stop them?
    // Reuse existing logic from previous file roughly
    if (!isDrawing) {
        path.on('click', (event) => {
            if (['select', 'move', 'edit'].includes(window.polygonManager.mode)) {
                event.stopPropagation();
                // Pass Transformed Coordinates?
                // `d3.pointer(event, g.node())` returns coords inside the group (Data Space)
                const [mx, my] = d3.pointer(event, g.node());
                window.polygonManager.selectPolygonAt(mx, my);
                renderPolygons();
                renderCanvas(); // Update highlights
            }
        });

        // Move drag logic... (kept same as before, assuming it uses d3.pointer on g)
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
                    renderCanvas();
                })
                .on('end', () => window.polygonManager.endMove())
            );
        }
    }

    // Vertices handles
    if (mode === 'edit' && isSelected) {
        const handleRadius = 6 / transform.k;
        vertices.forEach((v, i) => {
            group.append('circle')
                .attr('class', 'polygon-vertex')
                .attr('cx', v.x)
                .attr('cy', v.y)
                .attr('r', handleRadius)
                .call(d3.drag()
                    .on('drag', (event) => {
                        const [x, y] = d3.pointer(event, g.node());
                        if (window.polygonManager.moveVertex(i, x, y)) {
                            renderPolygons();
                            renderCanvas();
                        }
                    })
                    .on('end', () => {
                        if (window.updateStats) window.updateStats();
                    })
                );
        });
    }
}

function setupInteraction() {
    // Mouse move on SVG to detect points
    svg.on('mousemove', (event) => {
        // Get mouse position relative to SVG (Screen Space)
        const [sx, sy] = d3.pointer(event, svg.node());

        // Convert to Data Space using inverted transform
        // transform.invert is available on d3.zoomTransform objects usually.
        // But our 'transform' variable is a standard D3 transform object from event.transform
        const mx = (sx - transform.x) / transform.k;
        const my = (sy - transform.y) / transform.k;

        // Simple linear search for closest
        let closest = null;
        let minDesc = Infinity;
        // Selection Radius in Data Space
        const threshold = 15 / transform.k; // Increased threshold slightly

        for (const p of pointsData) {
            const dx = p.sx - mx;
            const dy = p.sy - my;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < threshold && dist < minDesc) {
                minDesc = dist;
                closest = p;
            }
        }

        if (hoveredPoint !== closest) {
            hoveredPoint = closest;
            canvas.style.cursor = closest ? 'pointer' : 'crosshair';
            renderCanvas();
        }
    });

    svg.on('click', (event) => {
        if (event.defaultPrevented) return;

        // Use hoveredPoint detected during mousemove
        if (hoveredPoint && window.polygonManager.mode === 'navigate') {
            event.stopPropagation();
            // Select point logic
            if (window.handlePointSelection) {
                window.handlePointSelection(hoveredPoint);
            }
            renderCanvas();
            return;
        }

        // Polygon Interaction coordinates
        // For polygons, we use the same math or rely on d3.pointer relative to g if reliable
        // Or manual:
        const [sx, sy] = d3.pointer(event, svg.node());
        const mx = (sx - transform.x) / transform.k;
        const my = (sy - transform.y) / transform.k;

        const mode = window.polygonManager.mode;

        if (mode === 'draw') {
            window.polygonManager.addVertex(mx, my);
            renderPolygons();
        } else if (mode === 'select') {
            // For polygon selection, we pass Data Coordinates
            window.polygonManager.selectPolygonAt(mx, my);
            renderPolygons();
            renderCanvas();
        } else if (mode === 'navigate') {
            // Deselect if clicking empty space
            if (window.handlePointSelection) window.handlePointSelection(null);
            renderCanvas();
        }
    });

    // Right click / Double click to close poly
    svg.on('contextmenu', (e) => {
        if (window.polygonManager.mode === 'draw') {
            e.preventDefault();
            window.polygonManager.closePolygon();
        }
    });
}

function resetVisualizationZoom() {
    if (!svg || !zoom) return;
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}

// Helpers
function highlightConnections(pointId) {
    // TODO: Implement Canvas line highlight logic
    // Usually easier to re-render renderCanvas with a "highlightedId" global
    // But for legacy support:
    // We can filter connectionsData in renderCanvas to draw specific ones thicker/brighter
    // Let's assume we redraw everything
    renderCanvas();
}

function resetHighlights() {
    renderCanvas();
}

function selectPointById(pointId) {
    const point = pointsData.find(p => p.id === pointId);
    if (!point) return;

    // Logic is handled by appState usually, but if we need visual update:
    // window.handlePointSelection usually updates appState which updates us
    // But here we can force redraw
    renderCanvas();

    if (window.handlePointSelection) {
        // This might cause loop if not careful? 
        // Usually this function is called FROM app.js responding to something else.
        // Just ensure visual update
    }
}

// Exports
window.initializeVisualization = initializeVisualization;
window.renderPolygons = renderPolygons;
window.renderPoints = renderPoints;
window.resetVisualizationZoom = resetVisualizationZoom;
window.highlightConnections = highlightConnections;
window.resetHighlights = resetHighlights;
window.selectPointById = selectPointById;
