// Main application state
const appState = {
    points: [],
    connections: [],
    polygons: [],
    selectedPoint: null,
    currentMode: 'navigate',
    isLoading: false,
    selectedFolderPath: null,
    neighborLinks: [] // To store current neighbor links data
};

// DOM Elements
const elements = {
    // New controls
    selectFolderBtn: document.getElementById('selectFolderBtn'),
    loadImagesBtn: document.getElementById('loadImagesBtn'),
    selectedFolder: document.getElementById('selectedFolder'),
    selectedFolder: document.getElementById('selectedFolder'),
    embeddingModel: document.getElementById('embeddingModel'),

    // Existing controls
    exportLabelsBtn: document.getElementById('exportLabelsBtn'),
    progressContainer: document.getElementById('progressContainer'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    statsInfo: document.getElementById('statsInfo'),
    rightPanel: document.getElementById('rightPanel'),
    closePanelBtn: document.getElementById('closePanelBtn'),
    imagePreview: document.getElementById('imagePreview'),
    imageName: document.getElementById('imageName'),
    imageLabel: document.getElementById('imageLabel'),
    currentMode: document.getElementById('currentMode'),

    // Tool buttons
    navigateBtn: document.getElementById('navigateBtn'),
    drawPolygonBtn: document.getElementById('drawPolygonBtn'),
    selectPolygonBtn: document.getElementById('selectPolygonBtn'),
    movePolygonBtn: document.getElementById('movePolygonBtn'),
    editPolygonBtn: document.getElementById('editPolygonBtn'),

    // Polygon info
    activePolygonInfo: document.getElementById('activePolygonInfo'),
    polygonNameInput: document.getElementById('polygonNameInput'),
    deletePolygonBtn: document.getElementById('deletePolygonBtn'),

    // Visualization
    resetZoomBtn: document.getElementById('resetZoomBtn'),

    // Modal
    imageModal: document.getElementById('imageModal'),
    expandedImage: document.getElementById('expandedImage'),
    caption: document.getElementById('caption'),
    closeModalSpan: document.querySelector('.close-modal')
};

// Event Listeners
elements.selectFolderBtn.addEventListener('click', handleSelectFolder);
elements.loadImagesBtn.addEventListener('click', handleLoadImages);
elements.exportLabelsBtn.addEventListener('click', handleExportLabels);
elements.closePanelBtn.addEventListener('click', closeRightPanel);
elements.resetZoomBtn.addEventListener('click', () => {
    if (window.resetVisualizationZoom) {
        window.resetVisualizationZoom();
    }
});

// Tool button listeners
elements.navigateBtn.addEventListener('click', () => setMode('navigate'));
elements.drawPolygonBtn.addEventListener('click', () => setMode('draw'));
elements.selectPolygonBtn.addEventListener('click', () => setMode('select'));
elements.movePolygonBtn.addEventListener('click', () => setMode('move'));
elements.editPolygonBtn.addEventListener('click', () => setMode('edit'));

// Polygon management listeners
elements.polygonNameInput.addEventListener('input', (e) => {
    if (window.polygonManager && window.polygonManager.selectedPolygon) {
        window.polygonManager.updatePolygonName(e.target.value);
        updateVisualization();
    }
});

elements.deletePolygonBtn.addEventListener('click', () => {
    if (window.polygonManager) {
        window.polygonManager.deleteSelectedPolygon();
        elements.activePolygonInfo.classList.add('hidden');
        updateVisualization();
        updateStats();
    }
});

// Modal Listeners
if (elements.closeModalSpan) {
    elements.closeModalSpan.onclick = function () {
        elements.imageModal.classList.add('hidden');
    }
}

window.onclick = function (event) {
    if (event.target == elements.imageModal) {
        elements.imageModal.classList.add('hidden');
    }
}

// Set application mode
function setMode(mode) {
    appState.currentMode = mode;

    // Update button states
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.classList.remove('active');
    });

    const modeButtons = {
        'navigate': elements.navigateBtn,
        'draw': elements.drawPolygonBtn,
        'select': elements.selectPolygonBtn,
        'move': elements.movePolygonBtn,
        'edit': elements.editPolygonBtn
    };
    if (modeButtons[mode]) {
        modeButtons[mode].classList.add('active');
    }

    // Update mode indicator
    const modeNames = {
        'navigate': 'Navegación',
        'draw': 'Dibujar Polígono',
        'select': 'Seleccionar Polígono',
        'move': 'Mover Polígono',
        'edit': 'Editar Polígono'
    };

    elements.currentMode.textContent = `Modo: ${modeNames[mode] || mode}`;

    // Update polygon manager mode
    if (window.polygonManager) {
        window.polygonManager.setMode(mode);
    }
}

// Handle folder selection dialog
async function handleSelectFolder() {
    try {
        elements.selectFolderBtn.disabled = true;
        elements.selectFolderBtn.textContent = 'Seleccionando...';

        const response = await fetch('/api/select-folder', { method: 'POST' });

        if (!response.ok) {
            throw new Error('Error abriendo diálogo de selección');
        }

        const data = await response.json();

        if (data.canceled) {
            elements.selectFolderBtn.innerHTML = '<span class="btn-icon">📁</span> Seleccionar Carpeta';
        } else {
            appState.selectedFolderPath = data.folder_path;
            elements.selectedFolder.textContent = `Carpeta: ${data.folder_path}`;
            elements.selectedFolder.classList.remove('hidden');
            elements.loadImagesBtn.classList.remove('hidden');
            elements.selectFolderBtn.innerHTML = '<span class="btn-icon">📁</span> Cambiar Carpeta';
        }
    } catch (error) {
        console.error('Error selecting folder:', error);
        alert(`Error: ${error.message}`);
    } finally {
        elements.selectFolderBtn.disabled = false;
        if (!appState.selectedFolderPath) {
            elements.selectFolderBtn.innerHTML = '<span class="btn-icon">📁</span> Seleccionar Carpeta';
        }
    }
}

// Handle image loading
async function handleLoadImages() {
    if (!appState.selectedFolderPath) {
        alert('Por favor, selecciona una carpeta primero');
        return;
    }

    try {
        appState.isLoading = true;
        elements.loadImagesBtn.disabled = true;
        elements.selectFolderBtn.disabled = true;
        elements.progressContainer.classList.remove('hidden');
        updateProgress(0, 'Iniciando proceso...');

        const response = await fetch('/api/load-images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                folder_path: appState.selectedFolderPath,
                model: elements.embeddingModel.value
            })
        });

        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            // Process all complete lines
            buffer = lines.pop(); // Keep last incomplete chunk in buffer

            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const msg = JSON.parse(line);

                    if (msg.type === 'progress') {
                        updateProgress(msg.value, msg.message);
                    } else if (msg.type === 'result') {
                        // Success
                        const data = msg.data;
                        appState.points = data.points;
                        appState.connections = data.connections;

                        updateProgress(100, 'Completado');

                        // Initialize visualization
                        if (window.initializeVisualization) {
                            window.initializeVisualization(appState.points, appState.connections);
                        }

                        // Enable export button
                        elements.exportLabelsBtn.disabled = false;
                        updateStats();

                        // Hide progress after delay
                        setTimeout(() => {
                            elements.progressContainer.classList.add('hidden');
                        }, 1500);

                    } else if (msg.type === 'error') {
                        throw new Error(msg.message);
                    }

                } catch (e) {
                    console.error('Error parsing stream message:', e);
                }
            }
        }

    } catch (error) {
        console.error('Error loading images:', error);
        alert(`Error: ${error.message}`);
        elements.progressContainer.classList.add('hidden');
    } finally {
        appState.isLoading = false;
        elements.loadImagesBtn.disabled = false;
        elements.selectFolderBtn.disabled = false;
    }
}

// Update progress bar
function updateProgress(percent, text) {
    elements.progressBar.style.width = `${percent}%`;
    elements.progressText.textContent = text || `${percent}%`;
}

// Update statistics
function updateStats() {
    const totalImages = appState.points.length;
    const totalPolygons = window.polygonManager ? window.polygonManager.polygons.length : 0;

    let labeledImages = 0;
    if (window.polygonManager) {
        window.polygonManager.polygons.forEach(polygon => {
            labeledImages += polygon.points.length;
        });
    }

    elements.statsInfo.innerHTML = `
        <div><strong>Total de imágenes:</strong> ${totalImages}</div>
        <div><strong>Polígonos creados:</strong> ${totalPolygons}</div>
        <div><strong>Imágenes etiquetadas:</strong> ${labeledImages}</div>
    `;
}

// Handle point selection
function handlePointSelection(point) {
    appState.selectedPoint = point;
    showImagePreview(point);

    // Highlight connections in visualization
    if (window.highlightConnections) {
        window.highlightConnections(point.id);
    }
}

// Show image preview
async function showImagePreview(point) {
    try {
        const response = await fetch(`/api/get-image/${point.id}`);
        const data = await response.json();

        // Update image
        elements.imagePreview.innerHTML = `<img src="${data.image}" alt="${data.filename}">`;

        // Setup Modal Trigger
        elements.imagePreview.onclick = function () {
            elements.imageModal.classList.remove('hidden');
            elements.expandedImage.src = data.image;
            elements.caption.innerHTML = `${point.filename} <br><small>${elements.imageLabel.textContent}</small>`;
        };

        // Update info
        elements.imageName.textContent = point.filename;

        // Get label from polygon
        let label = 'Sin etiqueta';
        if (window.polygonManager) {
            const polygon = window.polygonManager.getPolygonForPoint(point.id);
            if (polygon) {
                label = polygon.name || `Polígono ${polygon.id}`;
                elements.imageLabel.style.borderColor = polygon.color;
                elements.imageLabel.style.background = `${polygon.color}20`;
            } else {
                elements.imageLabel.style.borderColor = 'var(--border-color)';
                elements.imageLabel.style.background = 'rgba(99, 102, 241, 0.1)';
            }
        }
        elements.imageLabel.textContent = label;

        // Load Neighbors
        const neighborsList = document.getElementById('neighborsList');
        const neighborsContainer = document.getElementById('neighborsContainer');

        if (neighborsList && appState.connections) {
            neighborsList.innerHTML = '<div class="loading-text">Cargando vecinos...</div>';
            neighborsContainer.classList.remove('hidden');

            const neighborLinks = appState.connections.filter(c => c.source === point.id || c.source.id === point.id);
            const neighborIndices = neighborLinks.map(l => l.target);

            if (neighborIndices.length > 0) {
                neighborsList.innerHTML = '';

                // Fetch all neighbor images
                const neighborPromises = neighborIndices.map(async (idx) => {
                    try {
                        const resp = await fetch(`/api/get-image/${idx}`);
                        const imgData = await resp.json();
                        return { id: idx, ...imgData };
                    } catch (e) {
                        return null;
                    }
                });

                const neighbors = await Promise.all(neighborPromises);

                neighbors.forEach(n => {
                    if (n) {
                        const div = document.createElement('div');
                        div.className = 'neighbor-item';
                        div.innerHTML = `<img src="${n.image}" title="${n.filename}">`;
                        div.onclick = () => {
                            // Select point visually and logically
                            if (window.selectPointById) {
                                window.selectPointById(n.id);
                            }
                        };
                        neighborsList.appendChild(div);
                    }
                });
            } else {
                neighborsList.innerHTML = '<div class="no-neighbors">Sin vecinos conectados</div>';
            }
        }

        // Show panel
        elements.rightPanel.classList.remove('hidden');

    } catch (error) {
        console.error('Error loading image:', error);
    }
}

// Close right panel
function closeRightPanel() {
    elements.rightPanel.classList.add('hidden');
    appState.selectedPoint = null;

    // Deselect point in visualization
    if (window.deselectAllPoints) {
        window.deselectAllPoints(); // Might not be implemented, d3 logic handles click outside usually
        // Actually d3 click event handles clearing selection if logic exists
    }

    // Reset highlights
    if (window.resetHighlights) {
        window.resetHighlights();
    }

    // Reset point selection visual (if we want close button to clear everything)
    // We can just rely on user clicking elsewhere, but to be clean:
    // This requires calling d3 selection clearing again if selectPointById isn't enough.
    // For now, resetHighlights is the key requirement.
}

// Handle export labels
async function handleExportLabels() {
    try {
        // Get polygons data
        const polygonsData = window.polygonManager ? window.polygonManager.getPolygonsData() : [];

        const response = await fetch('/api/export-labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ polygons: polygonsData })
        });

        if (!response.ok) {
            throw new Error('Error al exportar etiquetas');
        }

        const data = await response.json();

        // Download CSV
        const blob = new Blob([data.csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        alert('Etiquetas exportadas correctamente');

    } catch (error) {
        console.error('Error exporting labels:', error);
        alert(`Error: ${error.message}`);
    }
}

// Update visualization (called from polygon manager)
function updateVisualization() {
    if (window.renderPolygons) {
        window.renderPolygons();
    }
}

// Show polygon info panel
function showPolygonInfo(polygon) {
    elements.activePolygonInfo.classList.remove('hidden');
    elements.polygonNameInput.value = polygon.name || '';
}

// Hide polygon info panel
function hidePolygonInfo() {
    elements.activePolygonInfo.classList.add('hidden');
}

// Export for other modules
window.appState = appState;
window.handlePointSelection = handlePointSelection;
window.updateStats = updateStats;
window.showPolygonInfo = showPolygonInfo;
window.hidePolygonInfo = hidePolygonInfo;
window.setMode = setMode;
