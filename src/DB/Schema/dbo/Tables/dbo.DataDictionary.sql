CREATE TABLE [dbo].[DataDictionary] (
    [ColumnID] INT NOT NULL IDENTITY(1,1)
    ,[ColumnName] VARCHAR(25) NOT NULL
    ,[Type] VARCHAR(25) NULL
    ,[Include] BIT NOT NULL
    ,[Unit] VARCHAR(10) NULL
    ,[Definition] VARCHAR(500) NULL
    ,CONSTRAINT [pk_DataDictionary] PRIMARY KEY ([ColumnID])
)
GO

