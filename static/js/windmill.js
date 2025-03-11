// Set up Three.js scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

// Initialize scene
function init() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('windmill-container').appendChild(renderer.domElement);

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Create windmill tower
    const towerGeometry = new THREE.CylinderGeometry(0.5, 1, 8, 12);
    const towerMaterial = new THREE.MeshPhongMaterial({ color: 0x666666 });
    const tower = new THREE.Mesh(towerGeometry, towerMaterial);
    scene.add(tower);

    // Create windmill hub
    const hubGeometry = new THREE.SphereGeometry(0.7, 32, 32);
    const hubMaterial = new THREE.MeshPhongMaterial({ color: 0x444444 });
    const hub = new THREE.Mesh(hubGeometry, hubMaterial);
    hub.position.y = 4;
    scene.add(hub);

    // Create windmill blades
    const bladesGroup = new THREE.Group();
    const bladeGeometry = new THREE.BoxGeometry(0.3, 4, 0.1);
    const bladeMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff });

    for (let i = 0; i < 3; i++) {
        const blade = new THREE.Mesh(bladeGeometry, bladeMaterial);
        blade.position.y = 2;
        blade.rotation.z = (i * Math.PI * 2) / 3;
        bladesGroup.add(blade);
    }

    bladesGroup.position.y = 4;
    scene.add(bladesGroup);

    // Position camera
    camera.position.z = 15;
    camera.position.y = 5;

    // Animation variables
    const animate = () => {
        requestAnimationFrame(animate);
        bladesGroup.rotation.z += 0.01;
        renderer.render(scene, camera);
    };

    animate();
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Sidebar toggle function
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (sidebar.style.width === '250px') {
        sidebar.style.width = '0';
        mainContent.style.marginLeft = '0';
    } else {
        sidebar.style.width = '250px';
        mainContent.style.marginLeft = '250px';
    }
} 