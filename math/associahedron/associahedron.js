const verts = [1, 1, 2, 5, 14, 42, 132]; //No reason to constantly calculate factorials.
const edges = [0, 0, 1, 5, 21, 84, 330];
let init = 5;
textBox(init)
function textBox(n) {
    // updates the graph info in left corner
    num_nodes = verts[n-1];
    num_links = edges[n-1];
    document.getElementById("graph-data").innerHTML = "Vertices: " + num_nodes.toString() + ". Edges: " + num_links.toString() + ".";
}
const Graph = ForceGraph3D()
    (document.getElementById('3d-graph'))
        .jsonUrl('./json/binary_words_of_'+ init.toString() + '.json') // Default associahedron
        .nodeLabel('id')
        .nodeLabel(node => node.name)
        .nodeThreeObject(function(){})
        .nodeRelSize(7)
        .nodeOpacity(1)
        .nodeColor(node => "rgb(114, 143, 255)")
        .nodeResolution(20)
        .linkOpacity(0.9)
        .linkWidth(0.3)  
        .linkColor(link=>"rgba(0,0,0,1)")
        .nodeLabel(label => `<span style= "color: rgb(0, 0, 0); lack">${label.name}</span>`)
        .backgroundColor("#ffffff")
        .onNodeDragEnd(node => { // Fix nodes after drag and drop
            node.fx = node.x;
            node.fy = node.y;
            node.fz = node.z;
        })
        .width(864)
        .height(500)
        .showNavInfo(false)
Graph.wordLen = init
Graph.d3Force('charge').strength(-500) // Choice of force + repulsion strength
// Define GUI to presentation of associahedron 
const settings = {
    K_n : init,
    showVertices: true,
    BinaryWords : false,
    Arrows : false,
};
const gui = new dat.GUI({autoPlace: false});
var gui_container = document.getElementById('gui');
gui_container.appendChild(gui.domElement);
const fileLoader = new THREE.FileLoader(); // for loading the .json
// Configure the GUI
const slider = gui.add(settings, 'K_n', 1, 7).step(1).name('K_n')
slider.onFinishChange( function updateAssociahedron(n) { // Upon change, change the .json file and resize the camera to fit new objects
    if(Graph.wordLen != n){
        Graph.wordLen = n;
        fileLoader.load(
        './json/binary_words_of_'+ n.toString() +'.json',
        // Callback for when .json file has finished loading
        function(jsonFile) {
            // Convert response text to JSON object
            const newAssociahedron = JSON.parse(jsonFile);
            // Pass resulting object to Graph
            Graph.graphData(newAssociahedron);
            vertexControl();
            textBox(n);
            zoomUpdate(n);
            document.getElementById("K_n").src = "./imgs/K_n/k_" + n.toString() + ".jpg";
            }
        );
    }
});

// Updating the zoom is a bit tricky. Here's a dumb workaround
function zoomUpdate(n){
    setTimeout(() => {
        if (n < 3){
            Graph.zoomToFit(1000, 300);
        }
        else if (n == 3){
            Graph.zoomToFit(1000, 200);
        }
        else{
            Graph.zoomToFit(1000, 100);
        }
    }, 250)
};

// Option to show or suppress parenthesizations
gui.add(settings, 'BinaryWords').name('Binary Words').onChange(()=>{vertexControl()}); 
// Option to show or suppress arrow tips
gui.add(settings, 'Arrows').onChange(() => {
    if(settings.Arrows){
        Graph.linkDirectionalArrowLength(5)
        Graph.linkDirectionalArrowRelPos(0.8)
    }
    else{
        Graph.linkDirectionalArrowLength(0)
        Graph.linkDirectionalArrowRelPos(0) 
    }
}); 
// We create an empty node once to avoid computing it over and over again
const obj = new THREE.Mesh(
    new THREE.SphereGeometry(7),
    new THREE.MeshBasicMaterial({ depthWrite: false, transparent: true, opacity: 0 })
);
// Option to show or suppress all vertices
gui.add(settings, 'showVertices').name('Show Vertices').onChange(() => {
    vertexControl()
}); 
// controls the presentation of the vertices
function vertexControl() {
    if(settings.showVertices){
        if(settings.BinaryWords){
            updateWords()
        }
        else{
            Graph.nodeThreeObject(()=>{});
        }
    }
    else{
        Graph.nodeThreeObject(()=>{return obj});
    }
}
// Creates the images if the user wants them. 
// This is done so as to not calculate the images over and over again; just once and recycle.
function makeImages(){
    nodes = Graph.graphData()["nodes"]
    if(!nodes[0].ownSpriteObj){
        nodes.forEach(node =>{
        item = node.img
        const imgObj = new THREE.Mesh(
        new THREE.SphereGeometry(7),
        new THREE.MeshBasicMaterial({ depthWrite: false, transparent: true, opacity: 0 })
        );
        // add img sprite as child
        const imgTexture = new THREE.TextureLoader().load(`./imgs/words/${item}`);
        const material = new THREE.SpriteMaterial({ map: imgTexture });
        const sprite = node.ownSpriteObj = new THREE.Sprite(material);
        // we need to fix the resolution on a case by case basis
        if(Graph.wordLen == 1){
            sprite.scale.set(10, 7);
        }
        else if(Graph.wordLen == 2){
            sprite.scale.set(15, 7);
        }
        else if(Graph.wordLen == 3){
            sprite.scale.set(30, 15);
        }
        else if(Graph.wordLen == 4){
            sprite.scale.set(50, 15);
        }
        else if(Graph.wordLen == 5){
            sprite.scale.set(100, 20);
        }
        else if(Graph.wordLen == 6){
            sprite.scale.set(120, 20);
        }
        else if (Graph.wordLen == 7){
            sprite.scale.set(140, 20);
        }
        })
    }
}   
// Updates the vertices to present the binary words
function updateWords(){
    makeImages(); // This will be ignored if the images are already made 
    Graph.nodeThreeObject(node => node.ownSpriteObj); //In any case, replace the nodes with the images
}