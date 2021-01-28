USE [MLBPitchClassification]
GO


SELECT
	*
FROM
	[dbo].[DataDictionary]
WHERE
	[Include] = 1
ORDER BY
	[ColumnName]


SELECT
	*
INTO
	#LesterSummByPark
FROM
	[dbo].[MLBPitch_2019_SummByPark]
WHERE
	[PitcherFullName] LIKE '%Lester%'


SELECT
	*
INTO
	#LesterSummByGame
FROM
	[dbo].[MLBPitch_2019_SummByGame]
WHERE
	[PitcherFullName] LIKE '%Lester%'


-- Plot values by park, i.e. How does velocity vary by park, release point vectors, etc...
SELECT * FROM #LesterSummByPark
SELECT * FROM #LesterSummByGame

