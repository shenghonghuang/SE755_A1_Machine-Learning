Attribute Information:

date time year-month-day hour:minute:second 
Temperature, in Celsius 
Relative Humidity, % 
Light, in Lux 
CO2, in ppm 
Humidity Ratio, Derived quantity from temperature and relative humidity, in kgwater-vapor/kg-air 
Occupancy, 0 or 1, 0 for not occupied, 1 for occupied status

Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.



Used Attributes:
TABLE XVIII. 	OCCUPANCY SENSOR USED ATTRIBUTES
Target	Occupancy
Features	Light
	CO2
Features Selection Justification:
       When people live, because they need to breathe, they change the carbon dioxide content and open the curtains or turn on the lights. The sensor gets a large change in light and carbon dioxide.

Unused Attributes:
      Temperature
      Humidity
      HumidityRatio
Features Selection Justification:
      Weather and season are the main reasons for changing Temperature, Humidity, and HumidityRatio. Human factors change little about these attributes.

