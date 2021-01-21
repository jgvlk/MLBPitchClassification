USE [MLBPitchClassification]
GO


TRUNCATE TABLE [dbo].[MLBPitch_2019]
GO


INSERT INTO [dbo].[MLBPitch_2019]
(
	[MLBGameID]
	,[PlayGUID]
	,[ID]
	,[EventNum]
	,[Des]
	,[DesEs]
	,[x0]
	,[y0]
	,[z0]
	,[vx0]
	,[vy0]
	,[vz0]
	,[ax0]
	,[ay0]
	,[az0]
	,[px]
	,[pz]
	,[pfx_x]
	,[pfx_z]
	,[Break_y]
	,[BreakAngle]
	,[BreakLength]
	,[StartSpeed]
	,[EndSpeed]
	,[Type]
	,[PitchType]
)
SELECT
	p.[MLBGameID]
	,[PlayGUID]
	,[ID]
	,[EventNum]
	,[Des]
	,[DesEs]
	,[x0]
	,[y0]
	,[z0]
	,[vx0]
	,[vy0]
	,[vz0]
	,[ax] AS [ax0]
	,[ay] AS [ay0]
	,[az] AS [az0]
	,[px]
	,[pz]
	,[pfx_x]
	,[pfx_z]
	,[Break_y]
	,[BreakAngle]
	,[BreakLength]
	,[StartSpeed]
	,[EndSpeed]
	,[Type]
	,[PitchType]
FROM
	[PitchFX].[stg].[Pitch] p
	JOIN [PitchFX].[stg].[Game] g ON p.[MLBGameID] = g.[MLBGameID]
WHERE
	YEAR(g.[Date]) = 2019 AND
	(
		p.[MLBGameID] IS NOT NULL AND
		[PlayGUID] IS NOT NULL AND
		[ID] IS NOT NULL AND
		[EventNum] IS NOT NULL AND
		[Des] IS NOT NULL AND
		[DesEs] IS NOT NULL AND
		[x0] IS NOT NULL AND
		[y0] IS NOT NULL AND
		[z0] IS NOT NULL AND
		[vx0] IS NOT NULL AND
		[vy0] IS NOT NULL AND
		[vz0] IS NOT NULL AND
		[ax] IS NOT NULL AND
		[ay] IS NOT NULL AND
		[az] IS NOT NULL AND
		[px] IS NOT NULL AND
		[pz] IS NOT NULL AND
		[pfx_x] IS NOT NULL AND
		[pfx_z] IS NOT NULL AND
		[Break_y] IS NOT NULL AND
		[BreakAngle] IS NOT NULL AND
		[BreakLength] IS NOT NULL AND
		[StartSpeed] IS NOT NULL AND
		[EndSpeed] IS NOT NULL AND
		[Type] IS NOT NULL AND
		[PitchType] IS NOT NULL
	)
ORDER BY
	p.[MLBGameID]
	,p.[ID]
	,p.[EventNum]

