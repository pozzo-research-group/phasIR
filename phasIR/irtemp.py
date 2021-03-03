# Function 1: Converts the raw centikelvin reading to Celcius
# Step: convert using given formula for centikelvin to celcius
# Input: centikelvin reading
# Output: float value in celcius
def centikelvin_to_celsius(temp):
    '''
    Converts given centikelvin value to Celsius

    Parameters
    -----------
    temp : Float
        The value of the temperature to be converted in centikelvin

    Returns
    --------
    cels : Float
        The converted value of the temperature in degree celcius
    '''
    cels = (temp - 27315)/100
    return cels


# Function: Converts raw centikelvin reading to fahrenheit
# Step:Use function (1) to convert to cels, use equation to convert to fahr
# Input: centikelvin reading
# Output: float value in fahrenheit
def to_fahrenheit(temp):
    '''
    Converts given centikelvin value to Celsius

    Parameters
    -----------
    temp : Float
        The value of the temperature to be converted in centikelvin

    Returns
    --------
    fahr : Float
        The converted value of the temperature in degree fahrenheit
    '''
    cels = centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return fahr


# Function: Covnerts raw centikelvin value to both fahrenheit and celcius
# Step: Use function (1) to convert to cels, use equation to convert to fahr
# Input: centikelvin reading
# Output: float values in celcius and fahrenheit
def to_temperature(temp):
    '''
    Converts given centikelvin value to both fahrenheit and celcius

    Parameters
    -----------
    temp : Float
        The value of the temperature to be converted in centikelvin

    Returns
    --------
    cels : Float
        The converted value of the temperature in degree celcius
    fahr : Float
        The converted value of the temperature in degree fahrenheit
    '''
    cels = centikelvin_to_celsius(temp)
    fahr = cels * 9 / 5 + 32
    return cels, fahr
