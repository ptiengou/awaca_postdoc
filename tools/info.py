sites=['ddu','d17','d47','d85','dmc']
sensors=['FLOWCAPT','SPC','SURF','WIND']#,'METEK']

prefix_to_resample_freq = {
    'FLOWCAPT':'10min',
    'SPC':'10min',
    'SURF':'10min',
    'WIND':'10min',
    'METEK':'100ms',
    'SURF_TABLE':'10min'
}

prefix_to_acquisition_freq = {
    'FLOWCAPT':'10min',
    'SPC':'1s',
    'SURF':'1min',
    'WIND':'1s',
    'METEK':'100ms',
    'SURF_WIND':'100ms'
}


variable_to_sensor = {
    # FLOWCAPT variables
    "RECORD": "FLOWCAPT",
    "FluxMin1": "FLOWCAPT",
    "FluxMean1": "FLOWCAPT",
    "FluxMax1": "FLOWCAPT",
    "FluxStd1": "FLOWCAPT",
    "FluxSum1": "FLOWCAPT",
    "FluxMin2": "FLOWCAPT",
    "FluxMean2": "FLOWCAPT",
    "FluxMax2": "FLOWCAPT",
    "FluxStd2": "FLOWCAPT",
    "FluxSum2": "FLOWCAPT",
    "WindMin2": "FLOWCAPT",
    "WindMean2": "FLOWCAPT",
    "WindMax2": "FLOWCAPT",
    "Hagl": "FLOWCAPT",
    "DT": "FLOWCAPT",

    # SPC variables
    "temperature": "SPC",
    "snowflux": "SPC",
    "particles_per_sec": "SPC",
    "voltage": "SPC",

    # SURF variables
    "T1_HMP_Avg": "SURF",
    "T1_HMP_Std": "SURF",
    "T2_HMP_Avg": "SURF",
    "T2_HMP_Std": "SURF",
    "T3_HMP_Avg": "SURF",
    "T3": "SURF",
    "T1": "SURF",
    "T2": "SURF",
    "pres": "SURF",
    "SWdn": "SURF",
    "SWin": "SURF",
    "wspd1": "SURF",
    "wspd2": "SURF",
    "wspd3": "SURF",
    "wspd3_Std": "SURF",
    "Batt_Volt": "SURF",
    "CR1X_Temp": "SURF",
    "wind_u_3_Min": "SURF",
    "wind_u_3_Max": "SURF",
    "wind_v_3_Avg": "SURF",
    "wind_v_3_Std": "SURF",
    "wind_v_3_Min": "SURF",
    "wind_v_3_Max": "SURF",
    "wind_v_2_Max": "SURF",
    "wdir2": "SURF",
    "RH1":"SURF",
    "RH2":"SURF",
    "RH3":"SURF",

    # WIND variables
    "wdir": "WIND",
    "wspd1": "WIND",
    "wspd2": "WIND",
    "wspd3": "WIND",
    "wdir1": "WIND",
    "wdir2": "WIND",
    "wspd1_Avg": "WIND",
    "wspd2_Avg": "WIND",
}

var_to_longname = {
    # FLOWCAPT variables
    "RECORD": "Record Number",
    "FluxMin1": "Minimum Snow Mass Flux (FlowCapt 1)",
    "FluxMean1": "Mean Snow Mass Flux (FlowCapt 1)",
    "FluxMax1": "Maximum Snow Mass Flux (FlowCapt 1)",
    "FluxStd1": "Standard Deviation of Snow Mass Flux (FlowCapt 1)",
    "FluxSum1": "Integrated Snow Mass Flux (FlowCapt 1)",
    "FluxMin2": "Minimum Snow Mass Flux (FlowCapt 2)",
    "FluxMean2": "Mean Snow Mass Flux (FlowCapt 2)",
    "FluxMax2": "Maximum Snow Mass Flux (FlowCapt 2)",
    "FluxStd2": "Standard Deviation of Snow Mass Flux (FlowCapt 2)",
    "FluxSum2": "Integrated Snow Mass Flux (FlowCapt 2)",
    "WindMin2": "Minimum Acoustic Wind Speed (FlowCapt 2)",
    "WindMean2": "Mean Acoustic Wind Speed (FlowCapt 2)",
    "WindMax2": "Maximum Acoustic Wind Speed (FlowCapt 2)",
    "Hagl": "Height Above Ground Level",
    "DT": "Sampling Time Step / Delta T",

    # SPC variables (Snow Particle Counter)
    "temperature": "Air Temperature",
    "snowflux": "Snow Mass Flux",
    "particles_per_sec": "Snow Particle Count per Second",
    "voltage": "Sensor Battery Voltage",

    # SURF variables
    "T1_HMP_Avg": "Mean Air Temperature Level 1",
    "T1_HMP_Std": "Standard Deviation of Air Temperature Level 1",
    "T2_HMP_Avg": "Mean Air Temperature Level 2",
    "T2_HMP_Std": "Standard Deviation of Air Temperature Level 2",
    "T3_HMP_Avg": "Mean Air Temperature Level 3",
    "T3": "Air Temperature Level 3",
    "T1": "Air Temperature Level 1",
    "T2": "Air Temperature Level 2",
    "pres": "Surface Atmospheric Pressure",
    "SWdn": "Downwelling Shortwave Radiation",
    "SWin": "Incoming Shortwave Radiation",
    "wspd1": "Wind Speed Level 1",
    "wspd2": "Wind Speed Level 2",
    "wspd3": "Wind Speed Level 3",
    "wspd3_Std": "Standard Deviation of Wind Speed Level 3",
    "Batt_Volt": "Datalogger Battery Voltage",
    "CR1X_Temp": "Datalogger Internal Temperature",
    "wind_u_3_Min": "Minimum U-Component Wind Speed Level 3",
    "wind_u_3_Max": "Maximum U-Component Wind Speed Level 3",
    "wind_v_3_Avg": "Mean V-Component Wind Speed Level 3",
    "wind_v_3_Std": "Standard Deviation of V-Component Wind Speed Level 3",
    "wind_v_3_Min": "Minimum V-Component Wind Speed Level 3",
    "wind_v_3_Max": "Maximum V-Component Wind Speed Level 3",
    "wind_v_2_Max": "Maximum V-Component Wind Speed Level 2",
    "wdir2": "Wind Direction Level 2",
    "RH1":"Relative humidity Level 1",
    "RH2":"Relative humidity Level 2",
    "RH3":"Relative humidity Level 3",

    # WIND variables
    "wdir": "Wind Direction",
    "wdir1": "Wind Direction Level 1",
    "wspd1_Avg": "Mean Wind Speed Level 1",
    "wspd2_Avg": "Mean Wind Speed Level 2",
}

var_to_units = {
    # FLOWCAPT variables
    "RECORD": "-",
    "FluxMin1": "g m⁻² s⁻¹",
    "FluxMean1": "g m⁻² s⁻¹",
    "FluxMax1": "g m⁻² s⁻¹",
    "FluxStd1": "g m⁻² s⁻¹",
    "FluxSum1": "g m⁻²",
    "FluxMin2": "g m⁻² s⁻¹",
    "FluxMean2": "g m⁻² s⁻¹",
    "FluxMax2": "g m⁻² s⁻¹",
    "FluxStd2": "g m⁻² s⁻¹",
    "FluxSum2": "g m⁻²",
    "WindMin2": "m s⁻¹",
    "WindMean2": "m s⁻¹",
    "WindMax2": "m s⁻¹",
    "Hagl": "m",
    "DT": "s",

    # SPC variables (Snow Particle Counter)
    "temperature": "0.1°C",
    "snowflux": "g m⁻² s⁻¹",
    "particles_per_sec": "s⁻¹",
    "voltage": "mV",

    # SURF variables
    "T1_HMP_Avg": "°C",
    "T1_HMP_Std": "°C",
    "T2_HMP_Avg": "°C",
    "T2_HMP_Std": "°C",
    "T3_HMP_Avg": "°C",
    "T3": "°C",
    "T1": "°C",
    "T2": "°C",
    "pres": "hPa",
    "SWdn": "W m⁻²",
    "SWin": "W m⁻²",
    "wspd1": "m s⁻¹",
    "wspd2": "m s⁻¹",
    "wspd3": "m s⁻¹",
    "wspd3_Std": "m s⁻¹",
    "Batt_Volt": "V",
    "CR1X_Temp": "°C",
    "wind_u_3_Min": "m s⁻¹",
    "wind_u_3_Max": "m s⁻¹",
    "wind_v_3_Avg": "m s⁻¹",
    "wind_v_3_Std": "m s⁻¹",
    "wind_v_3_Min": "m s⁻¹",
    "wind_v_3_Max": "m s⁻¹",
    "wind_v_2_Max": "m s⁻¹",
    "wdir2": "deg",

    # WIND variables
    "wdir": "deg",
    "wdir1": "deg",
    "wspd1_Avg": "m s⁻¹",
    "wspd2_Avg": "m s⁻¹",
}

var_to_maxval = {
    'snowflux'  : 300,
    'FluxMean1' : 300
}


