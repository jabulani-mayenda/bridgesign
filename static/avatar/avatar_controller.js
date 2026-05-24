import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

const ARM_MOTION_BONES = new Set([
    'hips',
    'spine',
    'chest',
    'upperChest',
    'neck',
    'head',
    'rightShoulder',
    'rightUpperArm',
    'rightLowerArm',
    'leftShoulder',
    'leftUpperArm',
    'leftLowerArm',
    'leftHand',
    'leftUpperLeg',
    'leftLowerLeg',
    'leftFoot',
    'rightUpperLeg',
    'rightLowerLeg',
    'rightFoot',
]);

const DISABLE_IDLE_ANIMATION = true;
const ENABLE_GENERATED_BODY_MOTION = true;
const QUATERNION_SMOOTHING = 0.42;
const RIGHT_HAND_PRESENTATION_EULER = new THREE.Euler(-0.35, 0.22, 0.72, 'XYZ');
const SIGN_PLAYBACK_SPEED = 1.28;
const GENERATED_SIGN_LABELS = new Set([
    'WHAT', 'WHERE', 'WHO', 'WHY', 'WHEN', 'HOW',
    'HELP', 'STOP', 'DANGER', 'EMERGENCY', 'FIRE', 'POLICE', 'AMBULANCE',
    'PAIN', 'SICK', 'HURT', 'SAD', 'TIRED', 'COLD',
    'HAPPY', 'LOVE', 'FAMILY', 'FRIEND',
    'ANGRY', 'BAD', 'FOOD', 'HOME', 'DOCTOR', 'NAME',
]);

const FINGER_BONES = {
    thumb: ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal'],
    index: ['IndexProximal', 'IndexIntermediate', 'IndexDistal'],
    middle: ['MiddleProximal', 'MiddleIntermediate', 'MiddleDistal'],
    ring: ['RingProximal', 'RingIntermediate', 'RingDistal'],
    little: ['LittleProximal', 'LittleIntermediate', 'LittleDistal'],
};

const HAND_SHAPES = {
    flat: {
        thumb: [0.22, 0.08, 0.04],
        index: [0.04, 0.03, 0.02],
        middle: [0.03, 0.02, 0.02],
        ring: [0.04, 0.03, 0.02],
        little: [0.06, 0.04, 0.03],
    },
    fist: {
        thumb: [0.95, 0.65, 0.32],
        index: [1.25, 1.35, 0.85],
        middle: [1.28, 1.38, 0.9],
        ring: [1.3, 1.4, 0.95],
        little: [1.34, 1.42, 1.0],
    },
    index: {
        thumb: [0.75, 0.5, 0.25],
        index: [0.02, 0.02, 0.02],
        middle: [1.18, 1.3, 0.9],
        ring: [1.22, 1.34, 0.92],
        little: [1.26, 1.38, 0.96],
    },
    w: {
        thumb: [0.86, 0.54, 0.28],
        index: [0.02, 0.02, 0.02],
        middle: [0.02, 0.02, 0.02],
        ring: [0.04, 0.03, 0.02],
        little: [1.18, 1.3, 0.92],
    },
    pinch: {
        thumb: [0.62, 0.38, 0.18],
        index: [0.22, 0.12, 0.04],
        middle: [0.22, 0.12, 0.04],
        ring: [1.2, 1.3, 0.9],
        little: [1.25, 1.36, 0.95],
    },
};

/**
 * BridgeSign VRM Avatar Controller
 * =================================
 * Loads a VRM humanoid model and drives sign language animations.
 * Compatible with @pixiv/three-vrm v2 + Three.js r160.
 */
export class AvatarController {
    constructor(containerId) {
        const container = document.getElementById(containerId);
        if (!container) throw new Error("Container not found: " + containerId);

        const w = container.clientWidth, h = container.clientHeight;
        if (w < 60 || h < 60) throw new Error(`Container too small: ${w}x${h}`);

        this.containerId = containerId;
        this.container = container;
        this.vrm = null;
        this.bones = {};
        this.animations = {};
        this.missingAnimations = new Set();
        this.animationQueue = [];
        this.isPlaying = false;
        this._currentAnim = null;
        this._bodyMotion = null;
        this._neutralPose = null;
        this._missingTrackBones = new Set();
        this._ready = false;
        this._loadFailed = false;

        /* ── Scene ─────────────────────────────────────── */
        this.scene = new THREE.Scene();

        const bgCanvas = document.createElement('canvas');
        bgCanvas.width = 2; bgCanvas.height = 512;
        const ctx = bgCanvas.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, 512);
        grad.addColorStop(0, '#2d3d3d');
        grad.addColorStop(0.5, '#1e2a2e');
        grad.addColorStop(1, '#141c22');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 2, 512);
        this.scene.background = new THREE.CanvasTexture(bgCanvas);

        /* ── Camera ────────────────────────────────────── */
        this.camera = new THREE.PerspectiveCamera(30, w / h, 0.1, 100);
        this.camera.position.set(0.0, 1.35, 3.2);
        this.camera.lookAt(0.0, 1.2, 0);

        /* ── Renderer ──────────────────────────────────── */
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        container.appendChild(this.renderer.domElement);

        /* ── Lighting ──────────────────────────────────── */
        this.scene.add(new THREE.AmbientLight(0xfff5ee, 0.55));
        const key = new THREE.DirectionalLight(0xffffff, 1.8);
        key.position.set(2, 3, 3);
        this.scene.add(key);
        const fill = new THREE.DirectionalLight(0xaaccee, 0.5);
        fill.position.set(-2, 1.5, 2);
        this.scene.add(fill);
        const rim = new THREE.DirectionalLight(0xffeedd, 0.6);
        rim.position.set(0, 2, -3);
        this.scene.add(rim);

        /* ── Clock ─────────────────────────────────────── */
        this.clock = new THREE.Clock();

        /* ── Loader text ───────────────────────────────── */
        const loaderEl = document.getElementById('avatarLoading');
        if (loaderEl) loaderEl.textContent = 'Loading VRM Avatar…';

        /* ── Render loop ───────────────────────────────── */
        this._animate = this._animate.bind(this);
        this._animate();
        window.addEventListener('resize', () => this.resize());

        /* ── Load model (async, doesn't block constructor) */
        this._readyPromise = this._loadVRM().catch(err => {
            console.error('[Avatar] VRM load failed:', err);
            this._loadFailed = true;
            const el = document.getElementById('avatarLoading');
            if (el) { el.textContent = 'Avatar model failed to load.'; el.style.display = 'block'; }
            window.dispatchEvent(new CustomEvent('avatar-error', {
                detail: { containerId: this.containerId, message: 'Avatar model failed to load.' }
            }));
            return null;
        });

        console.log('[Avatar] Controller created, loading VRM…');
    }

    /* ═══════════════════════════════════════════════════
       VRM Loading (three-vrm v2 compatible)
       ═══════════════════════════════════════════════════ */
    async _loadVRM() {
        const loader = new GLTFLoader();
        loader.register(parser => new VRMLoaderPlugin(parser));

        const gltf = await new Promise((resolve, reject) => {
            loader.load('/static/avatar/model/avatar.vrm', resolve,
                prog => {
                    const pct = prog.total > 0 ? Math.round(100 * prog.loaded / prog.total) : '…';
                    const el = document.getElementById('avatarLoading');
                    if (el) el.textContent = `Loading Avatar… ${pct}%`;
                },
                reject
            );
        });

        this.vrm = gltf.userData.vrm;
        if (!this.vrm) throw new Error('No VRM data in loaded model');

        // Optimise
        if (VRMUtils.removeUnnecessaryVertices) VRMUtils.removeUnnecessaryVertices(this.vrm.scene);
        if (VRMUtils.removeUnnecessaryJoints) VRMUtils.removeUnnecessaryJoints(this.vrm.scene);

        // This bundled VRM already faces the app camera.
        this.vrm.scene.rotation.y = 0;
        this._hideNonAvatarSceneExtras();
        this.scene.add(this.vrm.scene);

        // Cache bones
        this._cacheBones();
        this._captureNeutralPose();
        this._printAllBoneNames();

        // Set signing pose
        this._setSigningPose();

        // Hide loader
        const el = document.getElementById('avatarLoading');
        if (el) el.style.display = 'none';

        this._ready = true;
        console.log(`[Avatar] VRM loaded. ${Object.keys(this.bones).length} bones cached.`);
    }

    /* ═══════════════════════════════════════════════════
       Bone Cache — try both v2 APIs for compatibility
       ═══════════════════════════════════════════════════ */
    _cacheBones() {
        const humanoid = this.vrm.humanoid;
        if (!humanoid) { console.warn('[Avatar] No humanoid in VRM'); return; }

        // Helper: try getNormalizedBoneNode (VRM 1.0) then getBoneNode (VRM 0.x)
        const getBone = (name) => {
            if (humanoid.getNormalizedBoneNode) return humanoid.getNormalizedBoneNode(name);
            if (humanoid.getBoneNode) return humanoid.getBoneNode(name);
            // v2 fallback: try raw
            if (humanoid.getRawBoneNode) return humanoid.getRawBoneNode(name);
            return null;
        };

        const boneNames = [
            'hips',
            'rightHand',
            'rightThumbMetacarpal', 'rightThumbProximal', 'rightThumbDistal',
            'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
            'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
            'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
            'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
            'rightShoulder', 'rightUpperArm', 'rightLowerArm',
            'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
            'leftThumbMetacarpal', 'leftThumbProximal', 'leftThumbDistal',
            'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
            'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
            'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
            'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
            'spine', 'chest', 'upperChest', 'neck', 'head',
            'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
            'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
        ];

        for (const name of boneNames) {
            const node = getBone(name);
            if (node) this.bones[name] = node;
        }

        console.log(`[Avatar] Bones found: ${Object.keys(this.bones).join(', ')}`);
    }

    _printAllBoneNames() {
        const allBones = [];
        this.vrm.scene.traverse(node => {
            if (node.isBone) {
                allBones.push({
                    name: node.name || '(unnamed)',
                    parent: node.parent ? node.parent.name || '(unnamed parent)' : '(none)',
                });
            }
        });
        console.group('[Avatar] Full VRM skeleton bone names');
        console.table(allBones);
        console.groupEnd();

        const mapped = Object.entries(this.bones).map(([humanoidName, node]) => ({
            humanoidName,
            actualNodeName: node.name || '(unnamed)',
            parent: node.parent ? node.parent.name || '(unnamed parent)' : '(none)',
        }));
        console.group('[Avatar] Humanoid bone mapping used by BridgeSign');
        console.table(mapped);
        console.groupEnd();
    }

    _hideNonAvatarSceneExtras() {
        const hidden = [];
        this.vrm.scene.traverse(node => {
            const name = (node.name || '').toLowerCase();
            if (!node.visible) return;
            if (name.includes('robo_arm') || name.includes('roboarm')) {
                node.visible = false;
                hidden.push(node.name || '(unnamed)');
                return;
            }
            if (node.isBone || node.isSkinnedMesh || node.isMesh) return;
            if (name.includes('springbone') || name.includes('collider') || name.includes('gizmo') || name.includes('helper')) {
                node.visible = false;
                hidden.push(node.name || '(unnamed)');
            }
        });
        if (hidden.length) console.info(`[Avatar] Hidden non-render helper nodes: ${hidden.join(', ')}`);
    }

    _captureNeutralPose() {
        this._neutralPose = {};
        for (const [name, bone] of Object.entries(this.bones)) {
            this._neutralPose[name] = bone.quaternion.clone();
        }
    }

    /* ═══════════════════════════════════════════════════
       Signing Pose
       ═══════════════════════════════════════════════════ */
    _setSigningPose() {
        this._resetToNeutralPose();
        this._applyBodyPose(this._readyPose(), 1.0);
        this._setEmotion(null, 0);
    }

    _resetToNeutralPose() {
        if (!this._neutralPose) return;
        for (const [name, quat] of Object.entries(this._neutralPose)) {
            const bone = this.bones[name];
            if (bone) bone.quaternion.copy(quat);
        }
    }

    _idlePose(time = 0) {
        const breath = Math.sin(time * 1.8) * 0.035;
        const sway = Math.sin(time * 0.75) * 0.035;
        return {
            hips: [0.01 * Math.sin(time * 0.7), sway * 0.35, -sway * 0.25],
            spine: [0.03 + breath, sway * 0.3, sway * 0.35],
            chest: [0.05 + breath * 0.7, sway * 0.45, sway * 0.35],
            upperChest: [0.02 + breath * 0.4, sway * 0.25, sway * 0.2],
            neck: [-0.02, -sway * 0.25, -sway * 0.15],
            head: [-0.035 + breath * 0.18, -sway * 0.7, -sway * 0.25],
            rightShoulder: [0.01, -0.02, -0.05 + sway * 0.12],
            leftShoulder: [0.01, 0.02, 0.05 + sway * 0.12],
            rightUpperArm: [-0.26 + breath * 0.25, -0.08, -0.34 + sway * 0.18],
            rightLowerArm: [-0.34, 0.08, 0.1],
            rightHand: [0.02, 0.02, 0.02],
            leftUpperArm: [-0.24 + breath * 0.2, 0.08, 0.34 - sway * 0.18],
            leftLowerArm: [-0.28, -0.06, -0.08],
            leftHand: [0.02, -0.02, -0.02],
            leftUpperLeg: [0.0, 0.0, -0.02 + sway * 0.1],
            rightUpperLeg: [0.0, 0.0, 0.02 + sway * 0.1],
        };
    }

    _readyPose(time = this.clock ? this.clock.getElapsedTime() : 0) {
        const breath = Math.sin(time * 1.4) * 0.018;
        return {
            hips: [0.0, 0.0, 0.0],
            spine: [0.04 + breath, 0.0, 0.0],
            chest: [0.07 + breath, 0.0, 0.0],
            upperChest: [0.04 + breath * 0.5, 0.0, 0.0],
            neck: [-0.015, 0.0, 0.0],
            head: [-0.025, 0.0, 0.0],
            rightShoulder: [0.0, -0.02, -0.05],
            rightUpperArm: [-0.72, 0.08, -0.62],
            rightLowerArm: [-0.92, 0.08, 0.18],
            leftShoulder: [0.0, 0.02, 0.05],
            leftUpperArm: [-0.34, -0.04, 0.46],
            leftLowerArm: [-0.44, -0.04, -0.12],
        };
    }

    _pose(overrides = {}, time = 0) {
        const base = this._idlePose(time);
        for (const [boneName, value] of Object.entries(overrides)) {
            base[boneName] = value;
        }
        return base;
    }

    _lerp(a, b, t) {
        return a + (b - a) * t;
    }

    _smooth(t) {
        const v = Math.min(Math.max(t, 0), 1);
        return v * v * (3 - 2 * v);
    }

    _lerpPose(a, b, t) {
        const eased = this._smooth(t);
        const result = {};
        const names = new Set([...Object.keys(a), ...Object.keys(b)]);
        names.forEach(name => {
            const av = a[name] || [0, 0, 0];
            const bv = b[name] || av;
            result[name] = [
                this._lerp(av[0], bv[0], eased),
                this._lerp(av[1], bv[1], eased),
                this._lerp(av[2], bv[2], eased),
            ];
        });
        return result;
    }

    _relativeTargetQuaternion(boneName, localDelta) {
        const neutral = this._neutralPose && this._neutralPose[boneName];
        return neutral ? neutral.clone().multiply(localDelta).normalize() : localDelta.clone().normalize();
    }

    _applyBodyPose(pose, smoothing = QUATERNION_SMOOTHING) {
        for (const [boneName, rotation] of Object.entries(pose)) {
            const bone = this.bones[boneName];
            if (!bone) continue;
            const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(rotation[0], rotation[1], rotation[2], 'XYZ'));
            const target = this._relativeTargetQuaternion(boneName, delta);
            bone.quaternion.slerp(target, smoothing);
        }
    }

    _motionAccent(label) {
        const text = String(label || 'SIGN');
        let sum = 0;
        for (let i = 0; i < text.length; i++) sum += text.charCodeAt(i);
        return ((sum % 11) - 5) / 5;
    }

    _gesturePose(label, phase, duration) {
        const name = String(label || '').toUpperCase();
        const accent = this._motionAccent(name);
        const wave = Math.sin(phase * Math.PI * 2);
        const circleX = Math.cos(phase * Math.PI * 2);
        const circleY = Math.sin(phase * Math.PI * 2);
        const base = {
            hips: [0.0, accent * 0.02, -accent * 0.025],
            spine: [0.08, accent * 0.04, accent * 0.04],
            chest: [0.11, accent * 0.07, accent * 0.04],
            head: [-0.04, -accent * 0.1, accent * 0.035],
            rightShoulder: [0.04, -0.03 + accent * 0.02, -0.08],
            leftShoulder: [0.03, 0.03 - accent * 0.02, 0.08],
            rightUpperArm: [-0.54, 0.1 + accent * 0.05, -0.42 + accent * 0.1],
            rightLowerArm: [-0.62, 0.12, 0.16 + accent * 0.08],
            rightHand: [-0.18, 0.22 + accent * 0.04, 0.72],
            leftUpperArm: [-0.2, -0.06, 0.34],
            leftLowerArm: [-0.18, 0.04, -0.1],
            leftHand: [0.02, -0.04, -0.04],
        };

        if (['HELLO', 'GOODBYE'].includes(name)) {
            return {
                ...base,
                chest: [0.12, -0.08, -0.04],
                head: [-0.04, -0.16 + phase * 0.08, -0.07],
                rightUpperArm: [-1.22, -0.22 + phase * 0.2, -0.44 + wave * 0.16],
                rightLowerArm: [-0.88, 0.24, 0.46 + wave * 0.32],
                rightHand: [-0.18, 0.42, 1.0 + wave * 0.18],
                leftUpperArm: [-0.34, 0.12, 0.62],
            };
        }

        if (name === 'THANK_YOU') {
            return {
                ...base,
                hips: [0.05, 0.0, 0.0],
                spine: [0.16, 0.0, 0.0],
                chest: [0.2, 0.03, 0.0],
                head: [-0.13, 0.0, 0.0],
                rightUpperArm: [-1.22 + phase * 0.18, 0.12 + phase * 0.18, -0.38],
                rightLowerArm: [-0.98 + phase * 0.72, 0.28, 0.32],
                rightHand: [-0.38 + phase * 0.22, 0.34, 0.82],
            };
        }

        if (name === 'SORRY') {
            return {
                ...base,
                spine: [0.13, -0.03, 0.0],
                chest: [0.18, -0.04, 0.0],
                head: [-0.1, 0.03, 0.02],
                rightUpperArm: [-0.92 + circleY * 0.05, 0.4 + circleX * 0.08, -0.9 + circleY * 0.08],
                rightLowerArm: [-1.32 + circleX * 0.16, 0.04 + circleY * 0.1, -0.26 + circleX * 0.16],
                rightHand: [-0.34 + circleY * 0.18, 0.26 + circleX * 0.08, 0.54 + circleX * 0.18],
                leftUpperArm: [-0.36, -0.08, 0.48],
            };
        }

        if (name === 'PLEASE') {
            return {
                ...base,
                spine: [0.09, -0.05, 0.0],
                chest: [0.12, -0.06, 0.0],
                head: [-0.07, 0.05, 0.02],
                rightUpperArm: [-0.86 + circleY * 0.04, 0.34 + circleX * 0.08, -0.82],
                rightLowerArm: [-1.18 + circleX * 0.12, 0.06 + circleY * 0.08, -0.22 + circleY * 0.12],
                rightHand: [-0.22 + circleY * 0.12, 0.36, 0.36 + circleX * 0.14],
                leftUpperArm: [-0.36, -0.08, 0.5],
            };
        }

        if (name === 'YES') {
            return {
                ...base,
                head: [-0.16 + Math.sin(phase * Math.PI * 2) * 0.16, 0.0, 0.0],
                chest: [0.1, 0.0, 0.0],
                rightUpperArm: [-0.86, 0.14, -0.42],
                rightLowerArm: [-1.08 + wave * 0.22, 0.1, 0.2],
                rightHand: [-0.18 + wave * 0.18, 0.22, 0.58],
            };
        }

        if (name === 'NO') {
            return {
                ...base,
                head: [-0.04, Math.sin(phase * Math.PI * 2) * 0.24, 0.0],
                chest: [0.08, Math.sin(phase * Math.PI * 2) * 0.04, 0.0],
                rightUpperArm: [-0.88, 0.1, -0.36],
                rightLowerArm: [-0.9, 0.1 + wave * 0.16, 0.32],
                rightHand: [-0.2, 0.42 + wave * 0.2, 0.72],
            };
        }

        if (name === 'HELP') {
            return {
                ...base,
                spine: [0.12, 0.0, 0.0],
                chest: [0.18, 0.0, 0.0],
                rightUpperArm: [-0.96 - phase * 0.18, 0.18, -0.46],
                rightLowerArm: [-0.98 - phase * 0.18, 0.14, 0.24],
                rightHand: [-0.14, 0.16, 0.58],
                leftUpperArm: [-0.72 - phase * 0.14, -0.12, 0.44],
                leftLowerArm: [-0.92 - phase * 0.12, -0.08, -0.22],
                leftHand: [-0.1, -0.18, -0.58],
            };
        }

        if (['HELP', 'STOP', 'DANGER', 'EMERGENCY', 'FIRE', 'POLICE', 'AMBULANCE'].includes(name)) {
            const snap = name === 'STOP' ? 0.18 : 0.0;
            return {
                ...base,
                hips: [0.06 + snap, 0.0, -0.02],
                spine: [0.18 + snap, 0.02, 0.0],
                chest: [0.24 + snap, 0.02, 0.0],
                head: [-0.1, 0.0, 0.0],
                rightUpperArm: [-1.14, 0.08, -0.22],
                rightLowerArm: [-1.04, 0.18, 0.18],
                rightHand: [-0.22, 0.3, 0.75],
                leftUpperArm: [-0.66, -0.02, 0.28],
                leftLowerArm: [-0.72, -0.04, -0.12],
                leftHand: [-0.1, -0.25, -0.55],
            };
        }

        if (['PAIN', 'SICK', 'HURT', 'SAD', 'TIRED', 'COLD'].includes(name)) {
            return {
                ...base,
                hips: [-0.03, 0.0, accent * 0.02],
                spine: [-0.08, 0.0, accent * 0.04],
                chest: [-0.12, 0.0, accent * 0.04],
                head: [0.18, -accent * 0.05, 0.02],
                rightUpperArm: [-0.64, 0.16, -0.72],
                rightLowerArm: [-0.76, 0.08, 0.02],
                leftUpperArm: [-0.45, -0.08, 0.42],
            };
        }

        if (['HAPPY', 'LOVE', 'FAMILY', 'FRIEND'].includes(name)) {
            return {
                ...base,
                spine: [0.12, -accent * 0.03, 0.0],
                chest: [0.2, -accent * 0.04, 0.0],
                head: [-0.08, -accent * 0.08, -0.04],
                rightUpperArm: [-0.98, -0.02, -0.52],
                rightLowerArm: [-0.92, 0.1, 0.28],
                leftUpperArm: [-0.7, 0.06, 0.32],
                leftLowerArm: [-0.58, -0.04, -0.1],
            };
        }

        if (['ANGRY', 'BAD'].includes(name)) {
            return {
                ...base,
                hips: [0.03, 0.0, 0.0],
                spine: [0.18, 0.02, 0.0],
                chest: [0.24, 0.0, 0.0],
                head: [-0.18, 0.0, 0.0],
                rightUpperArm: [-0.82, 0.18, -0.34],
                rightLowerArm: [-1.08, 0.24, 0.22],
                leftUpperArm: [-0.5, -0.12, 0.46],
            };
        }

        if (name === 'WHAT') {
            return {
                ...base,
                spine: [0.08, 0.0, 0.0],
                chest: [0.12, 0.0, 0.0],
                head: [-0.05, wave * 0.06, 0.0],
                rightUpperArm: [-0.86, 0.22 + wave * 0.1, -0.58],
                rightLowerArm: [-0.78, 0.12, 0.38 + wave * 0.14],
                rightHand: [-0.42, 0.52, 0.92],
                leftUpperArm: [-0.86, -0.22 + wave * 0.1, 0.58],
                leftLowerArm: [-0.78, -0.12, -0.38 - wave * 0.14],
                leftHand: [-0.42, -0.52, -0.92],
            };
        }

        if (name === 'WHERE') {
            return {
                ...base,
                spine: [0.08, 0.0, 0.0],
                chest: [0.12, 0.0, 0.0],
                head: [-0.05, wave * 0.05, 0.0],
                rightUpperArm: [-0.92, 0.14 + wave * 0.1, -0.46],
                rightLowerArm: [-0.86, 0.12 + wave * 0.18, 0.26],
                rightHand: [-0.18, 0.5 + wave * 0.28, 0.82],
            };
        }

        if (name === 'WATER') {
            return {
                ...base,
                spine: [0.08, 0.0, 0.0],
                chest: [0.12, 0.0, 0.0],
                head: [-0.08, -0.02, 0.0],
                rightUpperArm: [-1.08, 0.18, -0.38],
                rightLowerArm: [-1.02 + wave * 0.08, 0.12, 0.24],
                rightHand: [-0.3 + wave * 0.14, 0.4, 0.76],
            };
        }

        if (['FOOD', 'HOME', 'DOCTOR', 'NAME', 'WHO', 'WHY', 'WHEN', 'HOW'].includes(name)) {
            return {
                ...base,
                spine: [0.08, accent * 0.04, 0.0],
                chest: [0.12, accent * 0.05, 0.0],
                head: [-0.06, -accent * 0.08, accent * 0.02],
                rightUpperArm: [-1.0, 0.16, -0.46],
                rightLowerArm: [-1.0, 0.14, 0.08],
                rightHand: [-0.24, 0.34, 0.76],
                leftUpperArm: [-0.3, -0.05, 0.48],
            };
        }

        if (/^[A-Z]$/.test(name)) {
            return {
                ...base,
                spine: [0.06, accent * 0.025, accent * 0.02],
                chest: [0.1, accent * 0.04, accent * 0.02],
                head: [-0.05, -accent * 0.08, 0.0],
                rightUpperArm: [-0.98, 0.12, -0.54 + accent * 0.06],
                rightLowerArm: [-0.94, 0.12, 0.2],
                rightHand: [-0.24, 0.34, 0.82],
            };
        }

        return base;
    }

    _createBodyMotion(label, duration) {
        const time = this.clock.getElapsedTime();
        const start = this._readyPose(time);
        const midA = this._gesturePose(label, 0.2, duration);
        const midB = this._gesturePose(label, 0.5, duration);
        const midC = this._gesturePose(label, 0.8, duration);
        const end = this._readyPose(time + duration);
        return {
            label: String(label || '').toUpperCase(),
            duration,
            emotion: this._emotionForLabel(label),
            keyframes: [
                { t: 0.0, pose: start },
                { t: 0.16, pose: midA },
                { t: 0.5, pose: midB },
                { t: 0.84, pose: midC },
                { t: 1.0, pose: end },
            ],
        };
    }

    _applyBodyMotion(motion, localTime) {
        if (!motion) return;
        const duration = Math.max(motion.duration || 1, 0.01);
        const progress = Math.min(Math.max(localTime / duration, 0), 1);
        const frames = motion.keyframes;
        let prev = frames[0];
        let next = frames[frames.length - 1];
        for (let i = 0; i < frames.length - 1; i++) {
            if (progress >= frames[i].t && progress <= frames[i + 1].t) {
                prev = frames[i];
                next = frames[i + 1];
                break;
            }
        }
        const span = Math.max(next.t - prev.t, 0.001);
        const pose = this._lerpPose(prev.pose, next.pose, (progress - prev.t) / span);
        const pulse = Math.sin(progress * Math.PI) * 0.04;
        if (pose.chest) pose.chest[0] += pulse;
        if (pose.head) pose.head[2] += Math.sin(progress * Math.PI * 2) * 0.025;
        if (pose.hips) pose.hips[2] += Math.sin(progress * Math.PI * 2) * 0.015;
        this._applyBodyPose(pose);
        this._setEmotion(motion.emotion, Math.sin(progress * Math.PI));
    }

    _updateIdleBody() {
        if (!DISABLE_IDLE_ANIMATION) {
            this._applyBodyPose(this._idlePose(this.clock.getElapsedTime()));
        } else {
            this._applyBodyPose(this._readyPose(), 0.12);
        }
        this._setEmotion(null, 0);
    }

    _emotionForLabel(label) {
        const name = String(label || '').toUpperCase();
        if (['HAPPY', 'LOVE', 'HELLO', 'GOODBYE', 'THANK_YOU', 'YES', 'FAMILY', 'FRIEND'].includes(name)) return 'happy';
        if (['SAD', 'PAIN', 'SICK', 'HURT', 'TIRED', 'COLD'].includes(name)) return 'sad';
        if (['ANGRY', 'DANGER', 'STOP', 'FIRE', 'EMERGENCY', 'BAD'].includes(name)) return 'angry';
        if (['PLEASE', 'SORRY', 'HELP'].includes(name)) return 'relaxed';
        return null;
    }

    _setEmotion(emotion, amount) {
        const manager = this.vrm && this.vrm.expressionManager;
        if (!manager || !manager.setValue) return;
        const names = ['happy', 'sad', 'angry', 'relaxed', 'surprised'];
        for (const name of names) manager.setValue(name, name === emotion ? Math.min(Math.max(amount * 0.75, 0), 1) : 0);
    }

    /* ═══════════════════════════════════════════════════
       Animation Loading & Playback
       ═══════════════════════════════════════════════════ */
    async _loadAnimation(label) {
        if (this.animations[label]) return this.animations[label];
        if (this.missingAnimations.has(label)) return null;
        try {
            const res = await fetch(`/static/avatar/animations/${label}.json`);
            if (!res.ok) throw new Error(res.status);
            const data = await res.json();
            this.animations[label] = data;
            return data;
        } catch {
            this.missingAnimations.add(label);
            return null;
        }
    }

    _createGeneratedClip(label) {
        const safeLabel = String(label || '').trim().toUpperCase();
        if (!GENERATED_SIGN_LABELS.has(safeLabel)) return null;
        return {
            name: safeLabel,
            duration: 1.15,
            source: 'generated_semantic_motion',
            tracks: [],
        };
    }

    _playClip(clipData) {
        if (!clipData) { this._playNext(); return; }
        if (!this._ready) {
            this.animationQueue.unshift(clipData);
            this.isPlaying = false;
            setTimeout(() => {
                if (!this.isPlaying) this._playNext();
            }, 120);
            return;
        }

        const tracks = {};
        for (const track of clipData.tracks) {
            const boneName = track.name.replace('.quaternion', '');
            const quats = [];
            for (let i = 0; i < track.values.length; i += 4) {
                quats.push(new THREE.Quaternion(
                    track.values[i], track.values[i+1], track.values[i+2], track.values[i+3]
                ));
            }
            tracks[boneName] = { times: track.times, quats };
        }
        const sourceDuration = clipData.duration || 1.0;
        const duration = Math.max(sourceDuration / SIGN_PLAYBACK_SPEED, 0.55);
        const hasRecordedArmMotion = Object.keys(tracks).some(name => ARM_MOTION_BONES.has(name));
        const bodyMotion = (!hasRecordedArmMotion && ENABLE_GENERATED_BODY_MOTION)
            ? this._createBodyMotion(clipData.name, duration)
            : null;

        this._currentAnim = {
            tracks, duration, sourceDuration,
            startTime: this.clock.getElapsedTime(),
            clipName: clipData.name,
            bodyMotion
        };

        const msg = document.getElementById('sttStatusText');
        if (msg) msg.innerText = `Signing: ${clipData.name}`;

        window.dispatchEvent(new CustomEvent('avatar-signing', {
            detail: { label: clipData.name, queueLength: this.animationQueue.length }
        }));
    }

    _updateAnimation() {
        if (!this._ready) return;
        if (!this._currentAnim) {
            this._updateIdleBody();
            return;
        }

        const { tracks, duration, sourceDuration, startTime, bodyMotion, clipName } = this._currentAnim;
        const t = this.clock.getElapsedTime() - startTime;

        if (t >= duration) {
            this._applyFrame(tracks, sourceDuration || duration);
            this._applyBodyMotion(bodyMotion, duration);
            this._applySemanticHandShape(clipName, 1.0);
            this._currentAnim = null;
            setTimeout(() => this._playNext(), 120);
            return;
        }
        const sourceTime = sourceDuration ? Math.min(t / duration, 1) * sourceDuration : t;
        this._applyFrame(tracks, sourceTime);
        this._applyBodyMotion(bodyMotion, t);
        this._applySemanticHandShape(clipName, 0.55);
    }

    _applyFrame(tracks, time) {
        const q = new THREE.Quaternion();
        for (const [boneName, { times, quats }] of Object.entries(tracks)) {
            const bone = this.bones[boneName];
            if (!bone) {
                if (!this._missingTrackBones.has(boneName)) {
                    this._missingTrackBones.add(boneName);
                    console.warn(`[Avatar] Animation track has no mapped bone: ${boneName}`);
                }
                continue;
            }

            let idx = 0;
            for (let i = 0; i < times.length - 1; i++) {
                if (time >= times[i]) idx = i;
            }
            const next = Math.min(idx + 1, times.length - 1);
            const t0 = times[idx], t1 = times[next];
            const alpha = t1 > t0 ? Math.min((time - t0) / (t1 - t0), 1.0) : 1.0;
            q.slerpQuaternions(quats[idx], quats[next], alpha);
            q.normalize();
            const localDelta = boneName === 'rightHand'
                ? q.clone().multiply(new THREE.Quaternion().setFromEuler(RIGHT_HAND_PRESENTATION_EULER))
                : q;
            const target = this._relativeTargetQuaternion(boneName, localDelta);
            bone.quaternion.slerp(target, QUATERNION_SMOOTHING);
        }
    }

    _handShapeForLabel(label) {
        const name = String(label || '').toUpperCase();
        if (['SORRY', 'YES'].includes(name)) return { right: 'fist' };
        if (['PLEASE', 'THANK_YOU', 'HELLO', 'GOODBYE'].includes(name)) return { right: 'flat' };
        if (name === 'WATER') return { right: 'w' };
        if (name === 'WHERE') return { right: 'index' };
        if (name === 'NO') return { right: 'pinch' };
        if (name === 'WHAT') return { right: 'flat', left: 'flat' };
        if (name === 'HELP') return { right: 'fist', left: 'flat' };
        if (name === 'STOP') return { right: 'flat', left: 'flat' };
        return null;
    }

    _applySemanticHandShape(label, smoothing = 0.55) {
        const shapeBySide = this._handShapeForLabel(label);
        if (!shapeBySide) return;
        for (const [side, shapeName] of Object.entries(shapeBySide)) {
            const shape = HAND_SHAPES[shapeName];
            if (!shape) continue;
            this._applyHandShape(side, shape, smoothing);
        }
    }

    _applyHandShape(side, shape, smoothing) {
        const prefix = side === 'left' ? 'left' : 'right';
        const spreadSign = side === 'left' ? -1 : 1;
        for (const [fingerName, curls] of Object.entries(shape)) {
            const bones = FINGER_BONES[fingerName];
            if (!bones) continue;
            for (let i = 0; i < bones.length; i++) {
                const boneName = `${prefix}${bones[i]}`;
                const bone = this.bones[boneName];
                if (!bone) continue;
                const spread = i === 0 && fingerName !== 'thumb'
                    ? spreadSign * this._fingerSpreadForShape(fingerName, shape)
                    : 0;
                const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(curls[i], 0, spread, 'XYZ'));
                const target = this._relativeTargetQuaternion(boneName, delta);
                bone.quaternion.slerp(target, smoothing);
            }
        }
    }

    _fingerSpreadForShape(fingerName, shape) {
        const isOpen = shape === HAND_SHAPES.flat || shape === HAND_SHAPES.w;
        if (!isOpen) return 0;
        if (fingerName === 'index') return -0.08;
        if (fingerName === 'ring') return 0.06;
        if (fingerName === 'little') return 0.12;
        return 0;
    }

    /* ═══════════════════════════════════════════════════
       Queue
       ═══════════════════════════════════════════════════ */
    async queueSign(signLabel) {
        if (!this._ready) {
            await this._readyPromise;
            if (!this._ready || this._loadFailed) return;
        }
        const label = String(signLabel || '').trim().toUpperCase();
        const clip = await this._loadAnimation(label) || this._createGeneratedClip(label);
        if (clip) {
            this.animationQueue.push(clip);
            if (!this.isPlaying) this._playNext();
        } else {
            const word = label.replace(/_/g, ' ').trim();
            if (word.length > 1) {
                const msg = document.getElementById('sttStatusText');
                if (msg) msg.innerText = `Spelling: ${word}`;
                await this.queueLetters(word);
            }
        }
    }

    async queueLetters(text) {
        for (const ch of String(text || '').replace(/[^A-Za-z]/g, '').toUpperCase()) {
            await this.queueSign(ch);
        }
    }

    async queueText(text) {
        const words = String(text || '').match(/[A-Za-z_]+/g) || [];
        for (const w of words) await this.queueSign(w.toUpperCase());
    }

    _playNext() {
        if (!this.animationQueue.length) {
            this.isPlaying = false;
            this._currentAnim = null;
            this._setSigningPose();
            window.dispatchEvent(new CustomEvent('avatar-idle'));
            return;
        }
        this.isPlaying = true;
        this._playClip(this.animationQueue.shift());
    }

    /* ═══════════════════════════════════════════════════
       Render
       ═══════════════════════════════════════════════════ */
    _animate() {
        requestAnimationFrame(this._animate);
        if (!this.renderer) return;
        this._updateAnimation();
        if (this.vrm) this.vrm.update(this.clock.getDelta());
        this.renderer.render(this.scene, this.camera);
    }

    resize() {
        if (!this.container || !this.camera || !this.renderer) return;
        const w = this.container.clientWidth, h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }
}

window.AvatarController = AvatarController;
console.log('[AvatarModule] VRM AvatarController (v2-compat) registered.');
