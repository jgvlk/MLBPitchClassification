USE [MLBPitchClassification]
GO


IF ( SELECT OBJECT_ID('[dbo].[MLBPitch_2019]') ) IS NOT NULL
	DROP TABLE [dbo].[MLBPitch_2019]
GO


SELECT
	ROW_NUMBER() OVER (ORDER BY [StartTFSZulu]) AS [ID]
	,p.[MLBGameID]
	,pm.[ParkID]
	,pm.[Name] AS [ParkName]
	,pm.[City] AS [ParkCity]
	,pm.[State] AS [ParkState]
	,pm.[League] AS [ParkLeague]
	,[PlayGUID_Pitch]
	,[PlayGUID_Event]
	,[Num]
	,[Des_Pitch]
	,[DesEs_Pitch]
	,[DesEs_Event]
	,[Des_Event]
	,[InningNum]
	,[InningHalf]
	,[EventEs]
	,[Event]
	,p.[AwayTeamRuns]
	,p.[HomeTeamRuns]
	,[Balls]
	,[Strikes]
	,[Outs]
	,[Pitcher] AS [PitcherID]
	,pit.[fullName] AS [PitcherFullName]
	,[PitcherThrows]
	,[Batter] AS [BatterID]
	,bat.[fullName] AS [BatterFullName] 
	,[BatterHeight]
	,[BatterStand]
	,[StartTFS]
	,[StartTFSZulu]
	,[EndTFSZulu]
	,[ax] AS [ax0]
	,[ay] AS [ay0]
	,[az] AS [az0]
	,[BreakAngle]
	,[BreakLength]
	,[Break_y]
	,[StartSpeed]
	,[EndSpeed]
	,[pfx_x]
	,[pfx_z]
	,[px]
	,[pz]
	,[SpinDir]
	,[SpinRate]
	,[sv_id]
	,[sz_bot]
	,[sz_top]
	,[vx0]
	,[vy0]
	,[vz0]
	,[x]
	,[x0]
	,[y]
	,[y0]
	,[z0]
	,[Zone]
	,[Nasty]
	,[PitchType]
	,[CC]
	,[Code]
	,[MT]
	,[TFS]
	,[TFSZulu]
	,[Type]
	,[TypeConfidence]
INTO
	[dbo].[MLBPitch_2019]
FROM
	[PitchFX].[stg].[vw_Pitch] p
	JOIN [PitchFX].[stg].[Game] g ON p.[MLBGameID] = g.[MLBGameID]
	JOIN [PitchFX].[stg].[Player] pit ON p.[Pitcher] = pit.[id]
	JOIN [PitchFX].[stg].[Player] bat ON p.[Batter] = bat.[id]
	LEFT JOIN [Retrosheet].[dbo].[udf_MLBGameIDConstructor]() gid ON g.[MLBGameID] = gid.[MLBGameID]
	LEFT JOIN [Retrosheet].[dbo].[Game] g2 ON gid.[GameID] = g2.[GameID]
	LEFT JOIN [Retrosheet].[dbo].[ParkMaster] pm ON g2.[ParkID] = pm.[ParkID]
WHERE
	YEAR(g.[Date]) = 2019 AND
	(
		p.[MLBGameID] IS NOT NULL AND
		[PlayGUID_Pitch] IS NOT NULL AND
		[PlayGUID_Event] IS NOT NULL AND
		[Num] IS NOT NULL AND
		[Des_Pitch] IS NOT NULL AND
		[DesEs_Pitch] IS NOT NULL AND
		[DesEs_Event] IS NOT NULL AND
		[Des_Event] IS NOT NULL AND
		[InningNum] IS NOT NULL AND
		[InningHalf] IS NOT NULL AND
		[EventEs] IS NOT NULL AND
		[Event] IS NOT NULL AND
		p.[AwayTeamRuns] IS NOT NULL AND
		p.[HomeTeamRuns] IS NOT NULL AND
		[Balls] IS NOT NULL AND
		[Strikes] IS NOT NULL AND
		[Outs] IS NOT NULL AND
		[Pitcher] IS NOT NULL AND
		[PitcherThrows] IS NOT NULL AND
		[Batter] IS NOT NULL AND
		[BatterHeight] IS NOT NULL AND
		[BatterStand] IS NOT NULL AND
		[StartTFS] IS NOT NULL AND
		[StartTFSZulu] IS NOT NULL AND
		[EndTFSZulu] IS NOT NULL AND
		[ax] IS NOT NULL AND
		[ay] IS NOT NULL AND
		[az] IS NOT NULL AND
		[BreakAngle] IS NOT NULL AND
		[BreakLength] IS NOT NULL AND
		[Break_y] IS NOT NULL AND
		[StartSpeed] IS NOT NULL AND
		[EndSpeed] IS NOT NULL AND
		[pfx_x] IS NOT NULL AND
		[pfx_z] IS NOT NULL AND
		[px] IS NOT NULL AND
		[pz] IS NOT NULL AND
		[SpinDir] IS NOT NULL AND
		[SpinRate] IS NOT NULL AND
		[sv_id] IS NOT NULL AND
		[sz_bot] IS NOT NULL AND
		[sz_top] IS NOT NULL AND
		[vx0] IS NOT NULL AND
		[vy0] IS NOT NULL AND
		[vz0] IS NOT NULL AND
		[x] IS NOT NULL AND
		[x0] IS NOT NULL AND
		[y] IS NOT NULL AND
		[y0] IS NOT NULL AND
		[z0] IS NOT NULL AND
		[Zone] IS NOT NULL AND
		[Nasty] IS NOT NULL AND
		[PitchType] IS NOT NULL AND
		[CC] IS NOT NULL AND
		[Code] IS NOT NULL AND
		[MT] IS NOT NULL AND
		[TFS] IS NOT NULL AND
		[TFSZulu] IS NOT NULL AND
		[Type] IS NOT NULL AND
		[TypeConfidence] IS NOT NULL
	)
ORDER BY
	[ID]
GO


CREATE CLUSTERED INDEX [ixc_MLBPitch_2019] ON [dbo].[MLBPitch_2019]([ID])
GO


IF ( SELECT OBJECT_ID('tempdb.dbo.#PitchFX') ) IS NOT NULL
    DROP TABLE #PitchFX
GO

SELECT
    [ID]
    ,[MLBGameID]
    ,[ParkID]
    ,[ParkName]
    ,[ParkCity]
    ,[ParkState]
    ,[ParkLeague]
    ,[PitcherID]
	,[PitcherFullName]
    ,[ax0]
    ,[ay0]
    ,[az0]
    ,[BreakAngle]
    ,[BreakLength]
    ,[Break_y]
    ,[StartSpeed]
    ,[EndSpeed]
    ,[pfx_x]
    ,[pfx_z]
    ,[px]
    ,[pz]
    ,[vx0]
    ,[vy0]
    ,[vz0]
    ,[x]
    ,[x0]
    ,[y]
    ,[y0]
    ,[z0]
INTO
    #PitchFX
FROM
    [dbo].[MLBPitch_2019]
ORDER BY
    [ID]
GO


IF ( SELECT OBJECT_ID('[dbo].[MLBPitch_2019_SummByPark]') ) IS NOT NULL
	DROP TABLE [dbo].[MLBPitch_2019_SummByPark]
GO

SELECT
	[PitcherID]
	,[ParkID]
	,[PitcherFullName]
    ,[ParkName]
    ,[ParkCity]
    ,[ParkState]
	,COUNT(*) AS [n]
    ,MIN([x0]) [x0_Min]
    ,MAX([x0]) [x0_Max]
    ,AVG([x0]) [x0_Avg]
    ,STDEVP([x0]) [x0_StDev]
    ,MIN([y0]) [y0_Min]
    ,MAX([y0]) [y0_Max]
    ,AVG([y0]) [y0_Avg]
    ,STDEVP([y0]) [y0_StDev]
    ,MIN([z0]) [z0_Min]
    ,MAX([z0]) [z0_Max]
    ,AVG([z0]) [z0_Avg]
    ,STDEVP([z0]) [z0_StDev]
    ,MIN([vx0]) [vx0_Min]
    ,MAX([vx0]) [vx0_Max]
    ,AVG([vx0]) [vx0_Avg]
    ,STDEVP([vx0]) [vx0_StDev]
    ,MIN([vy0]) [vy0_Min]
    ,MAX([vy0]) [vy0_Max]
    ,AVG([vy0]) [vy0_Avg]
    ,STDEVP([vy0]) [vy0_StDev]
    ,MIN([vz0]) [vz0_Min]
    ,MAX([vz0]) [vz0_Max]
    ,AVG([vz0]) [vz0_Avg]
    ,STDEVP([vz0]) [vz0_StDev]
    ,MIN([ax0]) [ax_Min]
    ,MAX([ax0]) [ax_Max]
    ,AVG([ax0]) [ax_Avg]
    ,STDEVP([ax0]) [ax_StDev]
    ,MIN([ay0]) [ay_Min]
    ,MAX([ay0]) [ay_Max]
    ,AVG([ay0]) [ay_Avg]
    ,STDEVP([ay0]) [ay_StDev]
    ,MIN([az0]) [az_Min]
    ,MAX([az0]) [az_Max]
    ,AVG([az0]) [az_Avg]
    ,STDEVP([az0]) [az_StDev]
    ,MIN([px]) [px_Min]
    ,MAX([px]) [px_Max]
    ,AVG([px]) [px_Avg]
    ,STDEVP([px]) [px_StDev]
    ,MIN([pz]) [pz_Min]
    ,MAX([pz]) [pz_Max]
    ,AVG([pz]) [pz_Avg]
    ,STDEVP([pz]) [pz_StDev]
    ,MIN([BreakAngle]) [BreakAngle_Min]
    ,MAX([BreakAngle]) [BreakAngle_Max]
    ,AVG([BreakAngle]) [BreakAngle_Avg]
    ,STDEVP([BreakAngle]) [BreakAngle_StDev]
    ,MIN([BreakLength]) [BreakLength_Min]
    ,MAX([BreakLength]) [BreakLength_Max]
    ,AVG([BreakLength]) [BreakLength_Avg]
    ,STDEVP([BreakLength]) [BreakLength_StDev]
    ,MIN([Break_y]) [Break_y_Min]
    ,MAX([Break_y]) [Break_y_Max]
    ,AVG([Break_y]) [Break_y_Avg]
    ,STDEVP([Break_y]) [Break_y_StDev]
    ,MIN([pfx_x]) [pfx_x_Min]
    ,MAX([pfx_x]) [pfx_x_Max]
    ,AVG([pfx_x]) [pfx_x_Avg]
    ,STDEVP([pfx_x]) [pfx_x_StDev]
    ,MIN([pfx_z]) [pfx_z_Min]
    ,MAX([pfx_z]) [pfx_z_Max]
    ,AVG([pfx_z]) [pfx_z_Avg]
    ,STDEVP([pfx_z]) [pfx_z_StDev]
    ,MIN([StartSpeed]) [StartSpeed_Min]
    ,MAX([StartSpeed]) [StartSpeed_Max]
    ,AVG([StartSpeed]) [StartSpeed_Avg]
    ,STDEVP([StartSpeed]) [StartSpeed_StDev]
    ,MIN([EndSpeed]) [EndSpeed_Min]
    ,MAX([EndSpeed]) [EndSpeed_Max]
    ,AVG([EndSpeed]) [EndSpeed_Avg]
    ,STDEVP([EndSpeed]) [EndSpeed_StDev]
INTO
	[dbo].[MLBPitch_2019_SummByPark]
FROM
    #PitchFX
GROUP BY
	[PitcherID]
	,[ParkID]
	,[PitcherFullName]
    ,[ParkName]
    ,[ParkCity]
    ,[ParkState]
ORDER BY
	[PitcherFullName]
	,[ParkName]
GO


IF ( SELECT OBJECT_ID('[dbo].[MLBPitch_2019_SummByGame]') ) IS NOT NULL
	DROP TABLE [dbo].[MLBPitch_2019_SummByGame]
GO

SELECT
	[MLBGameID]
	,[PitcherID]
	,[ParkID]
	,[PitcherFullName]
    ,[ParkName]
    ,[ParkCity]
    ,[ParkState]
	,COUNT(*) AS [n]
    ,MIN([x0]) [x0_Min]
    ,MAX([x0]) [x0_Max]
    ,AVG([x0]) [x0_Avg]
    ,STDEV([x0]) [x0_StDev]
    ,MIN([y0]) [y0_Min]
    ,MAX([y0]) [y0_Max]
    ,AVG([y0]) [y0_Avg]
    ,STDEV([y0]) [y0_StDev]
    ,MIN([z0]) [z0_Min]
    ,MAX([z0]) [z0_Max]
    ,AVG([z0]) [z0_Avg]
    ,STDEV([z0]) [z0_StDev]
    ,MIN([vx0]) [vx0_Min]
    ,MAX([vx0]) [vx0_Max]
    ,AVG([vx0]) [vx0_Avg]
    ,STDEV([vx0]) [vx0_StDev]
    ,MIN([vy0]) [vy0_Min]
    ,MAX([vy0]) [vy0_Max]
    ,AVG([vy0]) [vy0_Avg]
    ,STDEV([vy0]) [vy0_StDev]
    ,MIN([vz0]) [vz0_Min]
    ,MAX([vz0]) [vz0_Max]
    ,AVG([vz0]) [vz0_Avg]
    ,STDEV([vz0]) [vz0_StDev]
    ,MIN([ax0]) [ax_Min]
    ,MAX([ax0]) [ax_Max]
    ,AVG([ax0]) [ax_Avg]
    ,STDEV([ax0]) [ax_StDev]
    ,MIN([ay0]) [ay_Min]
    ,MAX([ay0]) [ay_Max]
    ,AVG([ay0]) [ay_Avg]
    ,STDEV([ay0]) [ay_StDev]
    ,MIN([az0]) [az_Min]
    ,MAX([az0]) [az_Max]
    ,AVG([az0]) [az_Avg]
    ,STDEV([az0]) [az_StDev]
    ,MIN([px]) [px_Min]
    ,MAX([px]) [px_Max]
    ,AVG([px]) [px_Avg]
    ,STDEV([px]) [px_StDev]
    ,MIN([pz]) [pz_Min]
    ,MAX([pz]) [pz_Max]
    ,AVG([pz]) [pz_Avg]
    ,STDEV([pz]) [pz_StDev]
    ,MIN([BreakAngle]) [BreakAngle_Min]
    ,MAX([BreakAngle]) [BreakAngle_Max]
    ,AVG([BreakAngle]) [BreakAngle_Avg]
    ,STDEV([BreakAngle]) [BreakAngle_StDev]
    ,MIN([BreakLength]) [BreakLength_Min]
    ,MAX([BreakLength]) [BreakLength_Max]
    ,AVG([BreakLength]) [BreakLength_Avg]
    ,STDEV([BreakLength]) [BreakLength_StDev]
    ,MIN([Break_y]) [Break_y_Min]
    ,MAX([Break_y]) [Break_y_Max]
    ,AVG([Break_y]) [Break_y_Avg]
    ,STDEV([Break_y]) [Break_y_StDev]
    ,MIN([pfx_x]) [pfx_x_Min]
    ,MAX([pfx_x]) [pfx_x_Max]
    ,AVG([pfx_x]) [pfx_x_Avg]
    ,STDEV([pfx_x]) [pfx_x_StDev]
    ,MIN([pfx_z]) [pfx_z_Min]
    ,MAX([pfx_z]) [pfx_z_Max]
    ,AVG([pfx_z]) [pfx_z_Avg]
    ,STDEV([pfx_z]) [pfx_z_StDev]
    ,MIN([StartSpeed]) [StartSpeed_Min]
    ,MAX([StartSpeed]) [StartSpeed_Max]
    ,AVG([StartSpeed]) [StartSpeed_Avg]
    ,STDEV([StartSpeed]) [StartSpeed_StDev]
    ,MIN([EndSpeed]) [EndSpeed_Min]
    ,MAX([EndSpeed]) [EndSpeed_Max]
    ,AVG([EndSpeed]) [EndSpeed_Avg]
    ,STDEV([EndSpeed]) [EndSpeed_StDev]
INTO
	[dbo].[MLBPitch_2019_SummByGame]
FROM
    #PitchFX
GROUP BY
	[MLBGameID]
	,[PitcherID]
	,[ParkID]
	,[PitcherFullName]
    ,[ParkName]
    ,[ParkCity]
    ,[ParkState]
ORDER BY
	[MLBGameID]
	,[PitcherFullName]
	,[ParkName]
GO



