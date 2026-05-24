import * as THREE from 'three';

/**
 * Manages the Three.js 3D Avatar scene — premium lighting & background
 */
export class AvatarScene {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) { console.error(`#${containerId} not found`); return; }

        this.scene = new THREE.Scene();

        // ── Gradient background ────────────────────────
        const bgCanvas = document.createElement('canvas');
        bgCanvas.width = 2; bgCanvas.height = 512;
        const bgCtx = bgCanvas.getContext('2d');
        const grad = bgCtx.createLinearGradient(0, 0, 0, 512);
        grad.addColorStop(0, '#2d3d3d');
        grad.addColorStop(0.5, '#1e2a2e');
        grad.addColorStop(1, '#141c22');
        bgCtx.fillStyle = grad;
        bgCtx.fillRect(0, 0, 2, 512);
        this.scene.background = new THREE.CanvasTexture(bgCanvas);

        // ── Camera (framing upper body + right hand) ───
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(38, aspect, 0.1, 100);
        this.camera.position.set(0.3, 1.45, 2.6);
        this.camera.lookAt(0.1, 1.15, 0);

        // ── Renderer ───────────────────────────────────
        try {
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.15;
            this.container.appendChild(this.renderer.domElement);
        } catch (err) {
            console.error("WebGL init failed:", err);
            this.renderer = null;
            return;
        }

        // ── Lighting (3-point) ─────────────────────────
        this.scene.add(new THREE.AmbientLight(0xfff5ee, 0.45));

        const key = new THREE.DirectionalLight(0xffffff, 1.6);
        key.position.set(2, 3, 3);
        this.scene.add(key);

        const fill = new THREE.DirectionalLight(0xaaccee, 0.5);
        fill.position.set(-2, 1.5, 2);
        this.scene.add(fill);

        const rim = new THREE.DirectionalLight(0xffeedd, 0.6);
        rim.position.set(0, 2, -3);
        this.scene.add(rim);

        // ── Animation mixer (set later by controller) ──
        this.mixer = null;
        this.clock = new THREE.Clock();
        this.avatar = null;

        window.addEventListener('resize', this.onWindowResize.bind(this));
        this.animate = this.animate.bind(this);
        this.animate();
    }

    onWindowResize() {
        if (!this.container || !this.camera || !this.renderer) return;
        const w = this.container.clientWidth, h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    animate() {
        requestAnimationFrame(this.animate);
        if (!this.renderer) return;
        if (this.mixer) this.mixer.update(this.clock.getDelta());
        this.renderer.render(this.scene, this.camera);
    }
}
