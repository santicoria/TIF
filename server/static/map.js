var map = L.map('map').setView([-36.5, -64], 4);

var actualEarthquakesGroup = L.featureGroup();
var predictedEarthquakesGroup = L.featureGroup();
var relationLines = L.featureGroup();

var collapseActualEarthquakesDiv = document.getElementById("collapseActualEarthquakes");
var collapseActualEarthquakesContent = collapseActualEarthquakesDiv.querySelector(".list-group");

var collapsePredictedEarthquakesDiv = document.getElementById("collapsePredictedEarthquakes");
var collapsePredictedEarthquakesContent = collapsePredictedEarthquakesDiv.querySelector(".list-group");

var collapseEarthquakePairsDiv = document.getElementById("collapseEarthquakesPairs");
var collapseEarthquakePairsContent = collapseEarthquakePairsDiv.querySelector(".list-group");

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

var study_area_latlngs = [
    [-55, -75],
    [-18, -75],
    [-18, -53],
    [-55, -53],
    [-55, -75],
];

var polyline_study_area = L.polyline(study_area_latlngs, {color: 'red', opacity: 0.2}).addTo(relationLines);

// Heatmap configuration
var cfg = {
    radius: 1.0,
    maxOpacity: 0.5,
    scaleRadius: true,
    useLocalExtrema: true,
    latField: 'lat',
    lngField: 'lng',
    valueField: 'count'
};

// Prepare heatmap data from actual and predicted earthquakes
var heatmapData = [];

// Add actual earthquakes (blue markers)
// var actualEarthquakes = {{ api_earthquakes|tojson }};
var actualEarthquakes = actualEarthquakesData;

actualEarthquakes.forEach(function(quake) {
    var earthquake_date = new Date(quake.time);
    const options = { month: "long" };
    heatmapData.push({ lat: quake.latitude, lng: quake.longitude, count: 1 });
    collapseActualEarthquakesContent.innerHTML += `
        <a href="#" id="actualEarthquakeCard-${(quake.id || 'N/A')}" class="list-group-item list-group-item-dark">
            <div class="d-flex w-100 justify-content-between">
                <h5 class="mb-1">Earthquake ${(quake.id || 'N/A')}</h5>
                <small>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" role="switch" id="switchCheckActualEarthquakeToggle-${(quake.id || 'N/A')}" checked>
                    </div>
                </small>
            </div>
            <p class="mb-1">
                <strong>Coordinates: </strong>${quake.coordinates}<br>
                <strong>Depth: </strong>${quake.depth} km<br>
                <strong>Magnitude: </strong>${quake.magnitude}
            </p>
            <small>${new Intl.DateTimeFormat("en-US", options).format(earthquake_date.getMonth())} ${earthquake_date.getDate().toString()} of ${earthquake_date.getFullYear().toString()}</small>
        </a>
    `;
    collapseEarthquakePairsContent.innerHTML += `
        <a href="#" id="earthquakePairCard-${(quake.id || 'N/A')}" class="list-group-item list-group-item-dark">
            <div class="d-flex w-100 justify-content-between">
                <h5 class="mb-1">Earthquake Pair ${(quake.id || 'N/A')}</h5>
                <small>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" role="switch" id="switchCheckEarthquakePairToggle-${(quake.id || 'N/A')}" checked>
                    </div>
                </small>
            </div>
        </a>
    `;
});

actualEarthquakes.forEach(function(quake) {
    var toggleThisEarthquakePair = document.getElementById('switchCheckEarthquakePairToggle-'+ quake.id);
    var earthquakePairCard = document.getElementById('earthquakePairCard-'+ quake.id);
    var earthquakeCard = document.getElementById('actualEarthquakeCard-'+ quake.id);
    var earthquake_date = new Date(quake.time);
    const options = { month: "long" };
    var month = new Intl.DateTimeFormat("en-US", options).format(earthquake_date)

    var eMarker = L.circle([quake.latitude, quake.longitude], {
        radius: Math.min(quake.magnitude * 10*1000, 100*1000),
        color: 'blue',
        fillColor: 'blue',
        fillOpacity: 0.2,
        id: (quake.id || 'N/A')
    }).addTo(actualEarthquakesGroup)
        .bindPopup("ID: " + (quake.id || 'N/A') + "<br>Coordinates: " + quake.coordinates + "<br>Depth: " + (quake.depth || 0) + " km<br>Magnitude: " + (quake.magnitude || 0) + "<br>Time: " + month.toString() + " " + earthquake_date.getDate().toString() + " of " + earthquake_date.getFullYear().toString() + " @ " + earthquake_date.getHours().toString()+":"+earthquake_date.getMinutes().toString()+":"+earthquake_date.getSeconds().toString())
        .on('click', function(e) {
            highlightSelected(quake.id);
        });
    var toggleThisAEarthquake = document.getElementById('switchCheckActualEarthquakeToggle-'+ quake.id);
    toggleThisAEarthquake.addEventListener('change', function(){
        if(map.hasLayer(eMarker)){
            if(earthquakePairCard.classList.contains("no-pe")) {
                toggleThisEarthquakePair.checked=false
                toggleThisEarthquakePair.dispatchEvent(new Event('change'));
            } else {
                map.removeLayer(eMarker);
                earthquakePairCard.classList.add("no-e")
            }   
        }
        else {
            map.addLayer(eMarker);
            earthquakePairCard.classList.remove("no-e")
            if(!earthquakePairCard.classList.contains("no-pe")) {
                toggleThisEarthquakePair.checked=true
            }
        }
    });

    /////////////////// FIX  TOGGLE WHEN PAIR IS DEACTIVATED ///////////////////

    toggleThisEarthquakePair.addEventListener('change', function(){
        if(map.hasLayer(eMarker) && earthquakePairCard.classList.contains("no-pe")){
            map.removeLayer(eMarker);
            toggleThisAEarthquake.checked=false
            earthquakePairCard.classList.add("no-e")
        } else if (map.hasLayer(eMarker) && !earthquakePairCard.classList.contains("no-pe")) {
            map.removeLayer(eMarker);
            toggleThisAEarthquake.checked=false
            earthquakePairCard.classList.add("no-e")
        } else if (!map.hasLayer(eMarker) && earthquakePairCard.classList.contains("no-pe")) {
            map.addLayer(eMarker);
            toggleThisAEarthquake.checked=true
            earthquakePairCard.classList.remove("no-e")
        } 
    });

});

map.addLayer(actualEarthquakesGroup);
// handle event of the group you choosed to be clicked
var toggleActualEarthquake = document.getElementById('switchCheckActualEarthquakeToggle');
toggleActualEarthquake.addEventListener('change', function(){
    // check if second group is on the map
    if(map.hasLayer(actualEarthquakesGroup)){
        map.removeLayer(actualEarthquakesGroup);
    }
    else {
        map.addLayer(actualEarthquakesGroup);
    }
});

// Add predicted earthquakes (red markers)
// var predictedEarthquakes = {{ predicted_earthquakes|tojson }};
var predictedEarthquakes = predictedEarthquakesData;

predictedEarthquakes.forEach(function(p_quake) {
    var earthquake_date = new Date(p_quake.predicted_time);
    const options = { month: "long" };
    heatmapData.push({ lat: p_quake.predicted_latitude, lng: p_quake.predicted_longitude, count: 1 });

    collapsePredictedEarthquakesContent.innerHTML += `
        <a href="#" id="predictedCard-${(p_quake.earthquake_id || 'N/A')}" class="list-group-item list-group-item-dark">
            <div class="d-flex w-100 justify-content-between">
                <h5 class="mb-1">Earthquake ${(p_quake.earthquake_id || 'N/A')}</h5>
                <small>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" role="switch" id="switchCheckPredictedEarthquakeToggle-${(p_quake.earthquake_id || 'N/A')}" checked>
                    </div>
                </small>
            </div>
            <p class="mb-1">
                <strong>Coordinates: </strong>${p_quake.predicted_coordinates}<br>
                <strong>Depth: </strong>${p_quake.predicted_depth} km<br>
                <strong>Magnitude: </strong>${p_quake.predicted_magnitude}
            </p>
            <small>${new Intl.DateTimeFormat("en-US", options).format(earthquake_date.getMonth())} ${earthquake_date.getDate().toString()} of ${earthquake_date.getFullYear().toString()}</small>
        </a>
    `;
});

// Initialize heatmap layer
var heatmapLayer = new HeatmapOverlay(cfg);
heatmapLayer.setData({ max: 10, data: heatmapData }); // Adjust max based on data density

// Toggle heatmap visibility
var heatmapVisible = false;
$('#toggleHeatmap').on('click', function() {
    if (heatmapVisible) {
        map.removeLayer(heatmapLayer);
        heatmapVisible = false;
    } else {
        heatmapLayer.addTo(map);
        heatmapVisible = true;
    }
});

predictedEarthquakes.forEach(function(p_quake) {
    // create a red polyline from an array of LatLng points
    var latlngs = [
        [p_quake.predicted_latitude, p_quake.predicted_longitude],
        [p_quake.latitude, p_quake.longitude]
    ];

    var polyline = L.polyline(latlngs, {color: 'red'}).addTo(relationLines);
    var pMarker = L.circle([p_quake.predicted_latitude, p_quake.predicted_longitude], {
        radius: 75*1000,
        color: 'red',
        fillColor: 'red',
        fillOpacity: 0.2
    }).addTo(predictedEarthquakesGroup)
        .bindPopup("ID: " + p_quake.earthquake_id + "<br>Coordinates: " + p_quake.coordinates + "<br>Depth: " + (p_quake.predicted_depth || 0) + " km<br>Magnitude: " + (p_quake.predicted_magnitude || 0) + "<br>Time: " + p_quake.predicted_time)
        .on('click', function(e) {
            highlightSelected(p_quake.earthquake_id);
        });

    var toggleThisEarthquakePair = document.getElementById('switchCheckEarthquakePairToggle-'+ p_quake.earthquake_id);
    var earthquakePairCard = document.getElementById('earthquakePairCard-'+ p_quake.earthquake_id);
    var toggleThisPEarthquake = document.getElementById('switchCheckPredictedEarthquakeToggle-' + p_quake.earthquake_id);
    toggleThisPEarthquake.addEventListener('change', function(){
        if(map.hasLayer(pMarker)){
            map.removeLayer(pMarker);
            earthquakePairCard.classList.add("no-pe")
            if(earthquakePairCard.classList.contains("no-e")) {
                toggleThisEarthquakePair.checked=false
                map.removeLayer(polyline);
            }
        }
        else {
            map.addLayer(pMarker);
            earthquakePairCard.classList.remove("no-pe")
            if(!earthquakePairCard.classList.contains("no-e")) {
                toggleThisEarthquakePair.checked=true
                map.addLayer(polyline);
            }
        }
    });

    toggleThisEarthquakePair.addEventListener('change', function(){
        if(map.hasLayer(pMarker) && earthquakePairCard.classList.contains("no-e")) {
            map.removeLayer(pMarker);
            map.removeLayer(polyline);
            toggleThisPEarthquake.checked=false
            earthquakePairCard.classList.add("no-pe")
        } else if (map.hasLayer(pMarker) && !earthquakePairCard.classList.contains("no-e")) {
            map.removeLayer(pMarker);
            map.removeLayer(polyline);
            toggleThisPEarthquake.checked=false
            earthquakePairCard.classList.add("no-pe")
        } else if (!map.hasLayer(pMarker) && !earthquakePairCard.classList.contains("no-e")) {
            map.addLayer(pMarker);
            map.addLayer(polyline);
            toggleThisPEarthquake.checked=true
            earthquakePairCard.classList.remove("no-pe")
        } else {
            map.removeLayer(polyline); 
        }

    });
});

map.addLayer(predictedEarthquakesGroup);
// handle event of the group you choosed to be clicked
var togglePredictedEarthquakes = document.getElementById('switchCheckPredictedEarthquakeToggle');
togglePredictedEarthquakes.addEventListener('change', function(){
    // check if second group is on the map
    if(map.hasLayer(predictedEarthquakesGroup)){
        map.removeLayer(predictedEarthquakesGroup);
    }
    else {
        map.addLayer(predictedEarthquakesGroup);
    }
});

map.addLayer(relationLines);
// handle event of the group you choosed to be clicked
var toggleRelationLines = document.getElementById('switchCheckRelationLines');
toggleRelationLines.addEventListener('change', function(){
    // check if second group is on the map
    if(map.hasLayer(relationLines)){
        map.removeLayer(relationLines);
    }
    else {
        map.addLayer(relationLines);
    }
});

window.addEventListener("DOMContentLoaded",e=>{
    // actualEarthquakes.forEach(function(quake) {
    //     var toggleThisAEarthquake = document.getElementById('switchCheckActualEarthquakeToggle-'+ quake.id);
    //     toggleThisAEarthquake.addEventListener('change', function(){
    //         // check if second group is on the map
    //         if(map.hasLayer(eMarker)){
    //             map.removeLayer(eMarker);
    //         }
    //         else {
    //             map.addLayer(eMarker);
    //         }
    //     });
    // });
});

function highlightSelected(e_id) {
    var predictedEarthquakesCards = collapsePredictedEarthquakesContent.querySelectorAll(".list-group-item");
    var actualEarthquakesCards = collapseActualEarthquakesContent.querySelectorAll(".list-group-item");
    var earthquakePairCards = collapseEarthquakePairsContent.querySelectorAll(".list-group-item");
    predictedEarthquakesCards.forEach(function(card) {
        card.classList.remove('active')
    });
    actualEarthquakesCards.forEach(function(card) {
        card.classList.remove('active')
    });
    earthquakePairCards.forEach(function(card) {
        card.classList.remove('active')
    });

    if(document.querySelector("#predictedCard-" + e_id)) {
        document.querySelector("#predictedCard-" + e_id).classList.toggle("active")
    }
    if(document.querySelector("#actualEarthquakeCard-" + e_id)) {
        document.querySelector("#actualEarthquakeCard-" + e_id).classList.toggle("active")
    }
    if(document.querySelector("#earthquakePairCard-" + e_id)) {
        document.querySelector("#earthquakePairCard-" + e_id).classList.toggle("active")
    }
}

jQuery(document).ready(function($) {
    
});