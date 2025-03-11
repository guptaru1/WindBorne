# WindBorne Insights - Agricultural & Renewable Energy Analytics

This project combines weather data from OpenMeteo with WindBorne's balloon data to provide valuable insights for agricultural operations and renewable energy optimization. The application uses AI to analyze the data and generate actionable recommendations.

## Features

1. **Agricultural Insights**
   - Optimal planting conditions based on weather patterns
   - Irrigation recommendations
   - Pest and disease risk assessment
   - Harvest timing suggestions

2. **Renewable Energy Optimization**
   - Solar energy potential analysis
   - Wind energy potential analysis
   - Grid optimization recommendations
   - Energy storage suggestions

3. **Real-time Weather & Balloon Data**
   - Live weather data from OpenMeteo
   - Balloon trajectory visualization
   - Atmospheric data analysis

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your OpenAI API key in the provided input field
2. The application will automatically fetch and display:
   - Agricultural insights with visualizations
   - Renewable energy optimization recommendations
   - Weather data and balloon trajectories
3. Data updates automatically every 5 minutes

## Technology Stack

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Data Visualization: Plotly
- AI: OpenAI GPT-4
- Weather Data: OpenMeteo API
- Balloon Data: WindBorne API

## Project Motivation

This project was created to help farmers make data-driven decisions and optimize renewable energy usage. By combining weather data with atmospheric measurements from WindBorne's balloons, we can provide more accurate and localized insights that help farmers improve crop yields and help energy providers optimize renewable energy generation and distribution.

## License

MIT License 