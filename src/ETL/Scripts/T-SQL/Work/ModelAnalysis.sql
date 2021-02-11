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

-- Cluster means
SELECT
	test.[Label_R]
	,AVG(p.[ax0]) [ax0]
	,AVG(p.[ay0]) [ay0]
	,AVG(p.[az0]) [az0]
	,AVG(p.[StartSpeed]) [StartSpeed]
	,AVG(p.[EndSpeed]) [EndSpeed]
	,AVG(p.[pfx_x]) [pfx_x]
	,AVG(p.[pfx_z]) [pfx_z]
	,AVG(p.[px]) [px]
	,AVG(p.[pz]) [pz]
	,AVG(p.[vx0]) [vx0]
	,AVG(p.[vy0]) [vy0]
	,AVG(p.[vz0]) [vz0]
	,AVG(p.[x0]) [x0]
	,AVG(p.[z0]) [z0]
FROM
	[output].[Model_v1_Test_R_Labeled] test
	JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
GROUP BY
	test.[Label_R]
ORDER BY
	test.[Label_R]


-- Population Means
SELECT
	AVG(p.[ax0]) [ax0]
	,AVG(p.[ay0]) [ay0]
	,AVG(p.[az0]) [az0]
	,AVG(p.[StartSpeed]) [StartSpeed]
	,AVG(p.[EndSpeed]) [EndSpeed]
	,AVG(p.[pfx_x]) [pfx_x]
	,AVG(p.[pfx_z]) [pfx_z]
	,AVG(p.[px]) [px]
	,AVG(p.[pz]) [pz]
	,AVG(p.[vx0]) [vx0]
	,AVG(p.[vy0]) [vy0]
	,AVG(p.[vz0]) [vz0]
	,AVG(p.[x0]) [x0]
	,AVG(p.[z0]) [z0]
FROM
	[dbo].[MLBPitch_2019] p
WHERE
	[PitcherThrows]	 = 'R'