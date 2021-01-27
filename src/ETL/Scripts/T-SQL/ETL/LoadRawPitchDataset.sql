USE [MLBPitchClassification]
GO


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
	,[Pitcher]
	,[PitcherThrows]
	,[Batter]
	,[BatterHeight]
	,[BatterStand]
	,[StartTFS]
	,[StartTFSZulu]
	,[EndTFSZulu]
	,[ax]
	,[ay]
	,[az]
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

