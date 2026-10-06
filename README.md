**Dynamic Asset Rating (DAR) Tool:**

The Dynamic Asset Rating (DAR) Tool is an interactive web application developed to estimate the time-varying loading capability of power transformers and overhead conductors under changing environmental conditions.
Conventional power system assets are often operated using fixed and conservative ratings. However, their actual thermal capacity depends on factors such as ambient temperature, wind speed, solar heating, equipment characteristics, and loading conditions. This application provides a practical way to evaluate the available capacity of these assets using dynamic thermal models.

The tool was developed by **Muhammad A. Barkhaya** as part of his research at Clarkson University’s Center for Electric Power Systems Research under **Prof. Dr Leo Yazhou Jiang.**

**Key Features**
- Dynamic rating calculation for power transformers
- Dynamic rating calculation for overhead transmission conductors
- Location-based weather data integration
- Interactive map and location selection
- User-defined equipment parameters
- Adjustable thermal and operating limits
- Seasonal and monthly rating analysis
- Statistical summaries and percentile-based visualization
- Comparison between static and dynamic asset ratings
- Browser-based interface with no local software installation required
  
**Transformer Dynamic Rating**
The transformer module is based on the thermal modeling framework of IEEE Std C57.91-2023.
The application evaluates transformer thermal behavior using parameters such as:
- Ambient temperature
- Transformer loading
- Top-oil temperature
- Winding hot-spot temperature
- Rated transformer capacity
- Cooling system
- Winding and core losses
- Thermal time constants
- Maximum allowable hot-spot temperature

Using these parameters, the tool determines the maximum loading level that can be applied without exceeding the selected thermal limit.
The resulting dynamic rating can be expressed in both per-unit and MVA, allowing direct comparison with the transformer nameplate rating.

**Dynamic Conductor Rating**
The conductor rating module follows the thermal principles of IEEE Std 738.
The conductor ampacity is determined by balancing:
- Joule heating
- Solar heating
- Convective cooling
- Radiative cooling
Environmental conditions such as ambient temperature and wind speed are used to determine how much electrical current the conductor can safely carry while remaining below its maximum operating temperature.
This allows the user to compare conventional static conductor ratings with dynamically calculated ampacity.
Weather Data

The application uses environmental data to calculate asset ratings for a selected geographic location.
Depending on the analysis, weather variables may include:
- Ambient air temperature
- Wind speed
- Solar conditions
- Geographic coordinates
- Time and seasonal variations

The integration of weather data allows the application to reflect how actual operating conditions influence the capacity of grid assets.

**Data Analysis and Visualization**

The application provides several visual and statistical outputs to help users interpret dynamic asset capability.
These include:
- Time-series plots
- Seasonal comparisons
- Monthly statistics
- Minimum, maximum, mean, and standard deviation
- Percentile distributions
- P10, median, and P90 values
- Dynamic rating versus nameplate/static rating
- Percentage increase or decrease in available capacity

These results help identify periods when additional capacity may be available and periods when environmental conditions reduce asset capability.
**Research Applications**

The application can be used for research and engineering studies involving:
- Dynamic Asset Rating
- Transmission capacity optimization
- Transformer asset management
- Dynamic Conductor Rating
- Grid congestion reduction
- Renewable energy integration
- Power system planning
- Climate-resilient grid operation
- Long-term asset utilization studies

The broader objective is to improve the utilization of existing power system infrastructure while maintaining equipment thermal limits and operational reliability.

**Technology Stack**

The application is primarily developed using:
Python
Streamlit
NumPy
Pandas
SciPy
Matplotlib
Plotly
Geospatial / Geocoding APIs
Weather Data APIs

**Code Structure**

A typical structure of the project can be described as:
DAR-Tool/
│
├── app.py
│   Main Streamlit application
│
├── transformer/
│   ├── thermal_model.py
│   ├── dtr_solver.py
│   └── transformer_parameters.py
│
├── conductor/
│   ├── ieee738_model.py
│   └── conductor_rating.py
│
├── weather/
│   ├── weather_data.py
│   └── geocoding.py
│
├── visualization/
│   ├── plots.py
│   └── statistics.py
│
├── data/
│   └── sample_transformer.csv
│
├── requirements.txt
└── README.md
