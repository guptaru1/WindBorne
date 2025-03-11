// Save API key to localStorage
function saveApiKey() {
    const apiKey = document.getElementById('apiKey').value;
    localStorage.setItem('openaiApiKey', apiKey);
    alert('API key saved successfully!');
}

// Fetch and display agricultural insights
async function fetchAgriculturalInsights() {
    try {
        const response = await fetch('/api/agricultural-insights');
        const data = await response.json();
        
        // Update agricultural chart
        const agriculturalChart = document.getElementById('agriculturalChart');
        Plotly.newPlot(agriculturalChart, data.chart_data, data.layout);
        
        // Update insights text
        const insightsDiv = document.getElementById('agriculturalInsights');
        insightsDiv.innerHTML = `
            <div class="insight-box">
                <h6>AI-Generated Insights</h6>
                <p>${data.insights}</p>
            </div>
        `;
    } catch (error) {
        console.error('Error fetching agricultural insights:', error);
    }
}

// Fetch and display renewable energy insights
async function fetchRenewableInsights() {
    try {
        const response = await fetch('/api/renewable-insights');
        const data = await response.json();
        
        // Update renewable chart
        const renewableChart = document.getElementById('renewableChart');
        Plotly.newPlot(renewableChart, data.chart_data, data.layout);
        
        // Update insights text
        const insightsDiv = document.getElementById('renewableInsights');
        insightsDiv.innerHTML = `
            <div class="insight-box">
                <h6>AI-Generated Insights</h6>
                <p>${data.insights}</p>
            </div>
        `;
    } catch (error) {
        console.error('Error fetching renewable insights:', error);
    }
}

// Fetch and display weather data
async function fetchWeatherData() {
    try {
        const response = await fetch('/api/weather-data');
        const data = await response.json();
        
        // Update weather chart
        const weatherChart = document.getElementById('weatherChart');
        Plotly.newPlot(weatherChart, data.chart_data, data.layout);
    } catch (error) {
        console.error('Error fetching weather data:', error);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Check if API key exists
    const apiKey = localStorage.getItem('openaiApiKey');
    if (apiKey) {
        document.getElementById('apiKey').value = apiKey;
    }
    
    // Initial data fetch
    fetchAgriculturalInsights();
    fetchRenewableInsights();
    fetchWeatherData();
    
    // Set up periodic updates
    setInterval(() => {
        fetchAgriculturalInsights();
        fetchRenewableInsights();
        fetchWeatherData();
    }, 300000); // Update every 5 minutes
});

// Handle scroll animations
window.addEventListener('scroll', () => {
    const windmill = document.getElementById('windmill-container');
    const buttons = document.querySelector('.buttons-container');
    const scrollPosition = window.scrollY;

    // Hide/show windmill based on scroll position
    if (scrollPosition > 100) {
        windmill.classList.add('hidden');
        buttons.classList.add('visible');
    } else {
        windmill.classList.remove('hidden');
        buttons.classList.remove('visible');
    }
}); 