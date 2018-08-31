# Regression
Used Attributes:
TABLE I. 	WORLD CUP USED ATTRIBUTES
Target	Total_Scores
Features	Team1_Attempts
	Team1_Corners
	Team1_Distance_Covered
	Team2_Attempts
	Team2_Corners
	Team2_Distance_Covered
Features Selection Justification:
     These attributes have higher correlation as Features. 

Unused Attributes 1:
Match_ID
Date
Location
Phase
Team1
Team1_Continent
Team2
Team2_Continent
Normal_Time
Features Selection Justification:
These columns only record text information such as date, time, country and location, and there is a speciality of recognition. Ignored these attributes ensures the generalisation of statistics. 

Unused Attributes 2:
Red_Card
Team1_Fouls
Team2_Yellow_Card
Team2_Red_Card
Team2_Fouls
Features Selection Justification:
These columns store non-positive information. 

Unused Attributes 3:
Team1_Offsides
Team1_Ball_Possession(%) 
Team1_Pass_Accuracy(%)
Team1_Distance_Covered
Team1_Ball_Recovered
Team2_Attempts
Team2_Corners
Team2_Offsides
Team2_Ball_Possession(%)
Team2_Pass_Accuracy(%)
Team2_Distance_Covered
Team2_Ball_Recovered
Features Selection Justification:
These attributes are non-critical data.

#Classification
Used Attributes:
TABLE VII. 	WORLD CUP USED ATTRIBUTES
Target	Match_result
Features	Team2_Yellow_Card
	Team2_Red_Card
	Team2_Fouls
Features Selection Justification:
      Through prediction, if Team2 has more fouls, the greater the probability of Team1 winning.
Unused Attributes 1:
Match_ID
Date
Location
Phase
Team1
Team1_Continent
Team2
Team2_Continent
Normal_Time
Features Selection Justification:
These columns only record text information such as date, time, country and location, and there is a speciality of recognition. Ignored these attributes ensures the generalisation of statistics. 

Unused Attributes 2:
Team1_Offsides
Team1_Ball_Possession(%) 
Team1_Pass_Accuracy(%)
Team1_Distance_Covered
Team1_Ball_Recovered
Team2_Attempts
Team2_Corners
Team2_Offsides
Team2_Ball_Possession(%)
Team2_Pass_Accuracy(%)
Team2_Distance_Covered
Team2_Ball_Recovered
Features Selection Justification:
These attributes are non-critical data.

Unused Attributes 3:
      Team1_Attempts
      Team1_Corners
      Team1_Distance_Covered
      Team2_Attempts
      Team2_Corners
      Team2_Distance_Covered
Features Selection Justification:
      These attributes are characterized by continuous behaviour and are not applicable to the study of classifier methods.	
Unused Attributes 4:
      Team1_Yellow_Card
      Team1_Red_Card
      Team1_Fouls
Features Selection Justification:
      Through prediction, if Team2 has more fouls, Team1 will have a higher probability of winning. However, Team1's foul data will conflict with the selected Team2 foul data.

