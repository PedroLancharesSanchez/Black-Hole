class PolygonManager {
    constructor() {
        this.polygons = [];
        this.currentPolygon = null; // Being drawn
        this.selectedPolygon = null;
        this.mode = 'navigate';
        this.colors = [
            '#00f0ff', '#7000ff', '#00ff9d', '#ff0055', '#ffe600',
            '#ff00ff', '#00ffff', '#ff8800', '#00ff00', '#ff00aa',
            '#ccff00', '#0099ff', '#aa00ff', '#ff3300', '#33ff00'
        ];
        this.colorIndex = 0;

        // Drag state
        this.dragStart = null;
        this.initialVertices = null;
    }

    setMode(mode) {
        this.mode = mode;
        // Reset creating state if leaving draw mode
        if (mode !== 'draw') {
            this.currentPolygon = null;
        }
        // Keep selection only if in move/edit/select
        if (!['select', 'move', 'edit'].includes(mode)) {
            this.selectedPolygon = null;
            if (window.hidePolygonInfo) window.hidePolygonInfo();
        }

        // Notify visualization to update cursor/state
        if (window.renderPolygons) window.renderPolygons();
    }

    // --- DRAWING ---

    startPolygon() {
        if (this.mode !== 'draw') return;
        this.currentPolygon = {
            id: Date.now(),
            vertices: [],
            points: [],
            color: this.getNextColor(),
            name: ''
        };
    }

    addVertex(x, y) {
        if (!this.currentPolygon) this.startPolygon();

        // Check if closing (near first point) - optional, handled by double click too
        if (this.currentPolygon.vertices.length > 2) {
            const first = this.currentPolygon.vertices[0];
            const dx = x - first.x;
            const dy = y - first.y;
            if (Math.hypot(dx, dy) < 15) {
                this.closePolygon();
                return;
            }
        }

        this.currentPolygon.vertices.push({ x, y });
    }

    closePolygon() {
        if (!this.currentPolygon || this.currentPolygon.vertices.length < 3) {
            alert("Need at least 3 points");
            return;
        }

        // Check overlaps
        if (this.checkOverlap(this.currentPolygon.vertices)) {
            alert("Polygon overlaps with an existing one!");
            this.currentPolygon = null;
            return;
        }

        // Name
        const name = prompt("Enter polygon name:", `Region ${this.polygons.length + 1}`);
        if (!name) {
            this.currentPolygon = null;
            return;
        }

        this.currentPolygon.name = name;
        this.currentPolygon.points = this.calculateContainedPoints(this.currentPolygon.vertices);

        this.polygons.push(this.currentPolygon);
        this.currentPolygon = null;

        if (window.updateStats) window.updateStats();
        if (window.renderPolygons) window.renderPolygons();
        if (window.renderPoints) window.renderPoints(); // Re-color points
    }

    // --- SELECTION ---

    selectPolygonAt(x, y) {
        // Find polygon containing point
        // Iterate backwards (topmost first)
        const found = this.polygons.slice().reverse().find(p =>
            this.isPointInside(x, y, p.vertices)
        );

        this.selectedPolygon = found || null;

        if (this.selectedPolygon && window.showPolygonInfo) {
            window.showPolygonInfo(this.selectedPolygon);
        } else if (window.hidePolygonInfo) {
            window.hidePolygonInfo();
        }

        return this.selectedPolygon;
    }

    // --- MOVING ---

    startMove(x, y) {
        if (!this.selectedPolygon) return false;
        this.dragStart = { x, y };
        this.initialVertices = JSON.parse(JSON.stringify(this.selectedPolygon.vertices));
        return true;
    }

    move(x, y) {
        if (!this.selectedPolygon || !this.dragStart) return;

        const dx = x - this.dragStart.x;
        const dy = y - this.dragStart.y;

        const newVertices = this.initialVertices.map(v => ({
            x: v.x + dx,
            y: v.y + dy
        }));

        // Check overlap (excluding self)
        if (this.checkOverlap(newVertices, this.selectedPolygon.id)) {
            return; // Block move if overlap
        }

        this.selectedPolygon.vertices = newVertices;
        // Update contained points
        this.selectedPolygon.points = this.calculateContainedPoints(newVertices);
    }

    endMove() {
        this.dragStart = null;
        this.initialVertices = null;
        if (window.renderPoints) window.renderPoints();
    }

    // --- EDITING ---

    moveVertex(vertexIndex, x, y) {
        if (!this.selectedPolygon) return;

        const original = this.selectedPolygon.vertices[vertexIndex];
        const newVertices = [...this.selectedPolygon.vertices];
        newVertices[vertexIndex] = { x, y };

        // Check overlap
        if (this.checkOverlap(newVertices, this.selectedPolygon.id)) {
            return false;
        }

        this.selectedPolygon.vertices = newVertices;
        this.selectedPolygon.points = this.calculateContainedPoints(newVertices);
        return true;
    }

    // --- UTILS ---

    getNextColor() {
        const c = this.colors[this.colorIndex % this.colors.length];
        this.colorIndex++;
        return c;
    }

    isPointInside(x, y, vertices) {
        let inside = false;
        for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
            const xi = vertices[i].x, yi = vertices[i].y;
            const xj = vertices[j].x, yj = vertices[j].y;
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    checkOverlap(vertices, excludeId = null) {
        // Simple check: check if any vertex of new poly is inside others
        // OR if any vertex of others is inside new poly
        // (Not perfect intersection check but good enough for UI)

        for (const p of this.polygons) {
            if (p.id === excludeId) continue;

            // Check new verts in old poly
            for (const v of vertices) {
                if (this.isPointInside(v.x, v.y, p.vertices)) return true;
            }
            // Check old verts in new poly
            for (const v of p.vertices) {
                if (this.isPointInside(v.x, v.y, vertices)) return true;
            }
        }
        return false;
    }

    calculateContainedPoints(vertices) {
        if (!window.appState || !window.appState.points) return [];
        const pointIds = [];
        window.appState.points.forEach(pt => {
            if (this.isPointInside(pt.screenX, pt.screenY, vertices)) {
                pointIds.push(pt.id);
            }
        });
        return pointIds;
    }

    // --- DATA ---

    updatePolygonName(name) {
        if (this.selectedPolygon) this.selectedPolygon.name = name;
    }

    deleteSelectedPolygon() {
        if (this.selectedPolygon) {
            this.polygons = this.polygons.filter(p => p.id !== this.selectedPolygon.id);
            this.selectedPolygon = null;
            if (window.renderPoints) window.renderPoints();
        }
    }

    getPolygonForPoint(pointId) {
        return this.polygons.find(p => p.points.includes(pointId));
    }

    getPolygonsData() {
        return this.polygons.map(p => ({
            name: p.name,
            points: p.points // Verify these are correct mapped IDs
        }));
    }
}

window.polygonManager = new PolygonManager();
