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


-- Population Mins
SELECT
	MIN(p.[ax0]) [ax0]
	,MIN(p.[ay0]) [ay0]
	,MIN(p.[az0]) [az0]
	,MIN(p.[StartSpeed]) [StartSpeed]
	,MIN(p.[EndSpeed]) [EndSpeed]
	,MIN(p.[pfx_x]) [pfx_x]
	,MIN(p.[pfx_z]) [pfx_z]
	,MIN(p.[px]) [px]
	,MIN(p.[pz]) [pz]
	,MIN(p.[vx0]) [vx0]
	,MIN(p.[vy0]) [vy0]
	,MIN(p.[vz0]) [vz0]
	,MIN(p.[x0]) [x0]
	,MIN(p.[z0]) [z0]
FROM
	[dbo].[MLBPitch_2019] p
WHERE
	[PitcherThrows]	 = 'R'


-- Population Maxs
SELECT
	MAX(p.[ax0]) [ax0]
	,MAX(p.[ay0]) [ay0]
	,MAX(p.[az0]) [az0]
	,MAX(p.[StartSpeed]) [StartSpeed]
	,MAX(p.[EndSpeed]) [EndSpeed]
	,MAX(p.[pfx_x]) [pfx_x]
	,MAX(p.[pfx_z]) [pfx_z]
	,MAX(p.[px]) [px]
	,MAX(p.[pz]) [pz]
	,MAX(p.[vx0]) [vx0]
	,MAX(p.[vy0]) [vy0]
	,MAX(p.[vz0]) [vz0]
	,MAX(p.[x0]) [x0]
	,MAX(p.[z0]) [z0]
FROM
	[dbo].[MLBPitch_2019] p
WHERE
	[PitcherThrows]	 = 'R'


-- Cluster Means
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
	[output].[Model_v3_Test_R_Labeled] test
	JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
GROUP BY
	test.[Label_R]
ORDER BY
	test.[Label_R]


-- Cluster Mins
SELECT
	test.[Label_R]
	,MIN(p.[ax0]) [ax0]
	,MIN(p.[ay0]) [ay0]
	,MIN(p.[az0]) [az0]
	,MIN(p.[StartSpeed]) [StartSpeed]
	,MIN(p.[EndSpeed]) [EndSpeed]
	,MIN(p.[pfx_x]) [pfx_x]
	,MIN(p.[pfx_z]) [pfx_z]
	,MIN(p.[px]) [px]
	,MIN(p.[pz]) [pz]
	,MIN(p.[vx0]) [vx0]
	,MIN(p.[vy0]) [vy0]
	,MIN(p.[vz0]) [vz0]
	,MIN(p.[x0]) [x0]
	,MIN(p.[z0]) [z0]
FROM
	[output].[Model_v3_Test_R_Labeled] test
	JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
GROUP BY
	test.[Label_R]
ORDER BY
	test.[Label_R]


-- Cluster Maxs
SELECT
	test.[Label_R]
	,MAX(p.[ax0]) [ax0]
	,MAX(p.[ay0]) [ay0]
	,MAX(p.[az0]) [az0]
	,MAX(p.[StartSpeed]) [StartSpeed]
	,MAX(p.[EndSpeed]) [EndSpeed]
	,MAX(p.[pfx_x]) [pfx_x]
	,MAX(p.[pfx_z]) [pfx_z]
	,MAX(p.[px]) [px]
	,MAX(p.[pz]) [pz]
	,MAX(p.[vx0]) [vx0]
	,MAX(p.[vy0]) [vy0]
	,MAX(p.[vz0]) [vz0]
	,MAX(p.[x0]) [x0]
	,MAX(p.[z0]) [z0]
FROM
	[output].[Model_v3_Test_R_Labeled] test
	JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
GROUP BY
	test.[Label_R]
ORDER BY
	test.[Label_R]

