USE [MLBPitchClassification]
GO


--SELECT * FROM [output].[Model_v1_Test_All_Labeled]
--SELECT * FROM [output].[Model_v1_Test_L_Labeled]
SELECT * FROM [output].[Model_v1_Test_R_Labeled]
--SELECT * FROM [output].[Model_v1_Train_All_Labeled]
--SELECT * FROM [output].[Model_v1_Train_L_Labeled]
SELECT * FROM [output].[Model_v1_Train_R_Labeled]

--SELECT COUNT(*) FROM [output].[Model_v1_Test_All_Labeled]
--SELECT COUNT(*) FROM [output].[Model_v1_Test_L_Labeled]
SELECT COUNT(*) FROM [output].[Model_v1_Test_R_Labeled] -- 159578
--SELECT COUNT(*) FROM [output].[Model_v1_Train_All_Labeled]
--SELECT COUNT(*) FROM [output].[Model_v1_Train_L_Labeled]

SELECT COUNT(*) FROM [output].[Model_v1_Train_R_Labeled] -- 372339





SELECT DISTINCT
	p.[ID]
	,p.[ax0]
	,p.[ay0]
	,p.[az0]
	,p.[StartSpeed]
	,p.[EndSpeed]
	,p.[pfx_x]
	,p.[pfx_z]
	,p.[px]
	,p.[pz]
	,p.[vx0]
	,p.[vy0]
	,p.[vz0]
	,p.[x0]
	,p.[z0]
	,test.[Label_All]
	,test.[Label_R]
	,test.[Label_L]
FROM
	[output].[Model_v1_Test_R_Labeled] test
	JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]


SELECT * FROM #LabeledPitchData_Test

