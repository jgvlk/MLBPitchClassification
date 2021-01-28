USE [MLBPitchClassification]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
IF ( SELECT OBJECT_ID('[dbo].[DataDictionary]') ) IS NOT NULL
	DROP TABLE [dbo].[DataDictionary]
GO
CREATE TABLE [dbo].[DataDictionary](
	[ColumnID] [int] IDENTITY(1,1) NOT NULL,
	[ColumnName] [varchar](25) NOT NULL,
	[Type] [varchar](25) NULL,
	[Include] [bit] NOT NULL,
	[Unit] [varchar](10) NULL,
	[Definition] [varchar](500) NULL,
 CONSTRAINT [pk_DataDictionary] PRIMARY KEY CLUSTERED 
(
	[ColumnID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
SET IDENTITY_INSERT [dbo].[DataDictionary] ON 

INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (1, N'des', N'result', 0, NULL, N'a brief text description of the result of the pitch')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (2, N'id', N'id', 0, NULL, N'a unique identification number per pitch within a game; numbers increment by one for each pitch but are not consecutive between at bats')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (3, N'type', N'result', 0, NULL, N'a one-letter abbreviation for the result of the pitch: B, ball; S strike (including fouls); X in play')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (4, N'x', NULL, 0, NULL, N'the horizontal location of the pitch as it crossed home plate as input by the Gameday stringer using the old Gameday coordinate system')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (5, N'y', NULL, 0, NULL, N'the vertical location of the pitch as it crossed home plate as input by the Gameday stringer using the old Gameday coordinate system')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (6, N'start_speed', N'speed', 1, N'mph', N'the pitch speed in three dimensions measured at the initial point y0')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (7, N'end_speed', N'speed', 1, N'mph', N'the pitch speed measured as it crossed the front of home plate')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (8, N'sz_top', N'3d-space', 1, N'ft', N'the distance from the ground to the top of the current batter''s rulebook strike zone as measured from the video by the PITCHf/x operator')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (9, N'sz_bot', N'3d-space', 1, N'ft', N'the distance from the ground to the bottom of the current batter''s rulebook strike zone; the PITCHf/x operator sets a line at the hollow of the knee for the bottom of the zone')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (10, N'pfx_x', N'movement', 1, N'in', N'the horizontal movement of the pitch between the release point and home plate as compared to a theoretical pitch thrown at the same speed with no spin-induced movement this parameter is measured at y=40 feet regardless of the y0 value')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (11, N'pfx_z', N'movement', 1, N'in', N'the vertical movement of the pitch between the release point and home plate as compared to a theoretical pitch thrown at the same speed with no spin-induced movement; this parameter is measured at y=40 feet regardless of the y0 value')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (12, N'px', N'3d-space', 1, N'ft', N'the left/right distance of the pitch from the middle of the plate as it crossed home plate; the PITCHf/x coordinate system is oriented to the catcher''s/umpire''s perspective with distances to the right being positive and to the left being negative')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (13, N'pz', N'3d-space', 1, N'ft', N'the height of the pitch as it crossed the front of home plate')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (14, N'x0', N'trajectory', 1, N'ft', N'the left/right distance of the pitch measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (15, N'y0', N'trajectory', 1, N'ft', N'the distance in feet from home plate where the PITCHf/x system is set to measure the initial parameters')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (16, N'z0', N'trajectory', 1, N'ft', N'the height of the pitch measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (17, N'vx0', N'trajectory', 1, N'ft/s', N'the velocity of the pitch in the x-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (18, N'vy0', N'trajectory', 1, N'ft/s', N'the velocity of the pitch in the y-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (19, N'vz0', N'trajectory', 1, N'ft/s', N'the velocity of the pitch in the z-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (20, N'ax0', N'trajectory', 1, N'ft/s^2', N'the acceleration of the pitch in the x-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (21, N'ay0', N'trajectory', 1, N'ft/s^2', N'the acceleration of the pitch in the y-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (22, N'az0', N'trajectory', 1, N'ft/s^2', N'the acceleration of the pitch in the z-direction measured at the initial point')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (23, N'break_y', N'trajectory', 1, N'ft', N'the distance from home plate to the point in the pitch trajectory where the pitch achieved its greatest deviation from the straight line path between the release point and the front of home plate')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (24, N'break_angle', N'trajectory', 1, N'deg', N'the angle from vertical to the straight line path from the release point to where the pitch crossed the front of home plate as seen from the catcher''s/umpire''s perspective')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (25, N'break_length', N'trajectory', 1, N'in', N'the measurement of the greatest distance between the trajectory of the pitch at any point between the release point and the front of home plate and the straight line path from the release point and the front of home plate')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (26, N'sv_id', N'id', 0, NULL, N'a date/time stamp of when the PITCHf/x tracking system first detected the pitch in the air; it is in the format YYMMDD_hhmmss')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (27, N'pitch_type', N'class', 0, NULL, N'the most probable pitch type according to a neural net classification algorithm developed by Ross Paul of MLBAM')
INSERT [dbo].[DataDictionary] ([ColumnID], [ColumnName], [Type], [Include], [Unit], [Definition]) VALUES (28, N'type_confidence', N'class', 0, NULL, N'the value of the weight at the classification algorithm''s output node corresponding to the most probable pitch type this value is multiplied by a factor of 1.5 if the pitch is known by MLBAM to be part of the pitcher''s repertoire')
SET IDENTITY_INSERT [dbo].[DataDictionary] OFF
GO
