' =========================================================================================
' ===           PRODUCTION-READY TRADE CONSOLIDATION PIPELINE                         ===
' ===           (Version 5.0 - Final, Hardened, and Validated)                        ===
' =========================================================================================
Option Explicit

Public Sub ConsolidateDailyTradeReports()

    ' 1. ================== CONFIGURATION & SETUP ==================
    Const SETTINGS_SHEET_NAME As String = "Settings"
    
    Dim sourcePath As String, filePattern As String, searchText As String
    Dim masterSheetName As String, errorLogSheetName As String, mismatchLogSheetName As String
    Dim headerRow As Long, dataStartRow As Long
    
    Dim fso As Object, wb As Workbook, wbSrc As Workbook
    Dim wsMaster As Worksheet, wsSrc As Worksheet, wsErr As Worksheet, wsMis As Worksheet
    Dim dictImported As Object, fname As String
    Dim processedCount As Long, totalRowsImported As Long
    Dim headerArr As Variant
    Dim lastRowSrc As Long, lastColSrc As Long, lastRowMaster As Long, logRow As Long
    Dim expectedRows As Long, importedRows As Long
    Dim i As Long

    ' Activate the top-level error handler immediately
    On Error GoTo FatalError

    Application.ScreenUpdating = False
    Application.EnableEvents = False
    Application.DisplayAlerts = False
    Application.Calculation = xlCalculationManual
    
    ' 2. ================== INITIALIZE & VALIDATE ENVIRONMENT ==================
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set wb = ThisWorkbook
    
    Call InitializeSettingsSheet(SETTINGS_SHEET_NAME)
    
    ' --- Load and Validate All Settings ---
    sourcePath = GetSetting(SETTINGS_SHEET_NAME, "SourceFolderPath", "H:\DESKTOP\DailyTradeReport\")
    filePattern = GetSetting(SETTINGS_SHEET_NAME, "FilePattern", "*DailyTradeMacro*.xlsx")
    searchText = GetSetting(SETTINGS_SHEET_NAME, "TradeSearchText", "EQTY EBKG")
    masterSheetName = GetSetting(SETTINGS_SHEET_NAME, "MasterSheetName", "consolidated trades")
    errorLogSheetName = GetSetting(SETTINGS_SHEET_NAME, "ErrorLogSheetName", "ErrorLog")
    mismatchLogSheetName = GetSetting(SETTINGS_SHEET_NAME, "MismatchLogSheetName", "MismatchLog")
    headerRow = CLng(GetSetting(SETTINGS_SHEET_NAME, "HeaderRowNumber", "3"))
    dataStartRow = CLng(GetSetting(SETTINGS_SHEET_NAME, "DataStartRowNumber", "4"))
    
    ' Validate Path Setting
    If Right(sourcePath, 1) <> "\" Then sourcePath = sourcePath & "\"
    
    If Not fso.FolderExists(sourcePath) Then
        MsgBox "Source folder not found, process cannot continue:" & vbCrLf & sourcePath, vbCritical, "Path Error"
        GoTo Cleanup
    End If
    
    ' Validate Row Number Settings
    If headerRow < 1 Or dataStartRow < 1 Or dataStartRow <= headerRow Then
        MsgBox "Invalid row numbers in Settings. Header must be before Data Start Row, and both must be > 0.", vbCritical, "Settings Error"
        GoTo Cleanup
    End If
    
    Set wsMaster = GetOrCreateSheet(masterSheetName, Array("SourceFile"))
    Set wsErr = GetOrCreateSheet(errorLogSheetName, Array("FileName", "ErrorDescription", "TimeLogged"))
    Set wsMis = GetOrCreateSheet(mismatchLogSheetName, Array("FileName", "ExpectedRows", "ImportedRows", "Comment"))
    
    If wsMaster Is Nothing Or wsErr Is Nothing Or wsMis Is Nothing Then GoTo Cleanup

    ' --- Build State Dictionary ---
    Set dictImported = CreateObject("Scripting.Dictionary")
    dictImported.CompareMode = vbTextCompare
    
    lastRowMaster = wsMaster.Cells(wsMaster.Rows.Count, "A").End(xlUp).Row
    If lastRowMaster >= 2 Then
        For i = 2 To lastRowMaster
            fname = Trim(wsMaster.Cells(i, "A").Value)
            If Len(fname) > 0 Then dictImported(fname) = True
        Next i
    End If
    
    ' 3. ================== PROCESS FILES ==================
    fname = Dir(sourcePath & filePattern)
    processedCount = 0
    totalRowsImported = 0
    
    Do While Len(fname) > 0
        If Not dictImported.Exists(fname) Then
            On Error GoTo FileError
            
            importedRows = 0
            Set wbSrc = Workbooks.Open(Filename:=sourcePath & fname, ReadOnly:=True, UpdateLinks:=False)
            Set wsSrc = wbSrc.Sheets(1)
            
            lastRowSrc = wsSrc.Cells(wsSrc.Rows.Count, "A").End(xlUp).Row
            ' --- Use loaded setting variables for all logic ---
            lastColSrc = wsSrc.Cells(headerRow, wsSrc.Columns.Count).End(xlToLeft).Column
            
            expectedRows = Application.WorksheetFunction.CountIf(wsSrc.Range("A" & dataStartRow & ":A" & lastRowSrc), "*" & searchText & "*")
            
            If lastRowSrc >= dataStartRow Then
                ' Get headers from the current file
                headerArr = wsSrc.Cells(headerRow, 1).Resize(1, lastColSrc).Value
                ' Ensure master sheet has correct headers before processing data
                Call EnsureMasterHeader(wsMaster, headerArr)

                For i = dataStartRow To lastRowSrc
                    If InStr(1, wsSrc.Cells(i, 1).Value, searchText, vbTextCompare) > 0 Then
                        lastRowMaster = wsMaster.Cells(wsMaster.Rows.Count, "A").End(xlUp).Row + 1
                        
                        wsMaster.Cells(lastRowMaster, "A").Value = fname
                        wsMaster.Cells(lastRowMaster, "B").Resize(1, lastColSrc).Value = wsSrc.Cells(i, 1).Resize(1, lastColSrc).Value
                        
                        importedRows = importedRows + 1
                    End If
                Next i
            End If
            
            wbSrc.Close SaveChanges:=False
            
            If importedRows > 0 Then
                dictImported.Add fname, True
                processedCount = processedCount + 1
                totalRowsImported = totalRowsImported + importedRows
            End If
            
            If expectedRows <> importedRows Then
                logRow = wsMis.Cells(wsMis.Rows.Count, "A").End(xlUp).Row + 1
                wsMis.Range("A" & logRow & ":D" & logRow).Value = Array(fname, expectedRows, importedRows, "Row count mismatch")
            End If
        End If
        
NextFile:
        fname = Dir
    Loop
    
' 4. ================== FINALISE & REPORT ==================
    If processedCount > 0 Then
        wsMaster.Columns.AutoFit
        MsgBox "Successfully consolidated " & processedCount & " new report(s)." & vbCrLf & _
               "A total of " & totalRowsImported & " trade rows were imported.", vbInformation, "Process Complete"
    Else
        MsgBox "No new reports found to consolidate.", vbInformation, "Up To Date"
    End If
    
' 5. ================== CLEANUP & EXIT ==================
Cleanup:
    Application.DisplayAlerts = True
    Application.EnableEvents = True
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Set fso = Nothing
    Set dictImported = Nothing
    Exit Sub
    
FileError:
    logRow = wsErr.Cells(wsErr.Rows.Count, "A").End(xlUp).Row + 1
    wsErr.Range("A" & logRow & ":C" & logRow).Value = Array(fname, "Error #" & Err.Number & ": " & Err.Description, Now)
    If Not wbSrc Is Nothing Then wbSrc.Close False
    Set wbSrc = Nothing
    Err.Clear
    Resume NextFile

FatalError:
    MsgBox "A fatal, unrecoverable error occurred:" & vbCrLf & vbCrLf & _
           "Error #" & Err.Number & " - " & Err.Description, vbCritical, "System Failure"
    Resume Cleanup
End Sub

Private Sub InitializeSettingsSheet(ByVal sheetName As String)
    Dim ws As Worksheet, wb As Workbook, sheetExists As Boolean
    Set wb = ThisWorkbook
    
    For Each ws In wb.Worksheets
        If LCase(ws.Name) = LCase(sheetName) Then sheetExists = True: Exit For
    Next ws

    If Not sheetExists Then
        Set ws = wb.Sheets.Add(After:=wb.Sheets(wb.Sheets.Count))
        ws.Name = sheetName
        
        With ws
            .Range("A1:C1").Value = Array("Setting Name", "Value", "Description")
            .Range("A1:C1").Font.Bold = True
            
            .Range("A2:C2").Value = Array("SourceFolderPath", "H:\DESKTOP\DailyTradeReport\", "The folder path for reports. Must end with a \")
            .Range("A3:C3").Value = Array("FilePattern", "*DailyTradeMacro*.xlsx", "The file name pattern. Use * as a wildcard.")
            .Range("A4:C4").Value = Array("TradeSearchText", "EQTY EBKG", "The key text to identify rows to import.")
            .Range("A5:C5").Value = Array("MasterSheetName", "consolidated trades", "Name of the main data consolidation sheet.")
            .Range("A6:C6").Value = Array("ErrorLogSheetName", "ErrorLog", "Name of the sheet for logging file errors.")
            .Range("A7:C7").Value = Array("MismatchLogSheetName", "MismatchLog", "Name of the sheet for logging data count mismatches.")
            .Range("A8:C8").Value = Array("HeaderRowNumber", 3, "The row number of the header in source files.")
            .Range("A9:C9").Value = Array("DataStartRowNumber", 4, "The row number where data starts in source files.")
            
            .Columns("A:C").AutoFit
        End With
    End If
End Sub

Private Function GetSetting(ByVal sheetName As String, ByVal settingKey As String, ByVal defaultValue As String) As String
    Dim settingsSheet As Worksheet, foundRow As Variant, result As String
    
    On Error Resume Next
    Set settingsSheet = ThisWorkbook.Sheets(sheetName)
    On Error GoTo 0
    
    If settingsSheet Is Nothing Then GetSetting = defaultValue: Exit Function
    
    foundRow = Application.Match(settingKey, settingsSheet.Columns(1), 0)
    
    If IsError(foundRow) Then
        result = defaultValue
    Else
        result = Trim(settingsSheet.Cells(foundRow, 2).Value)
        If result = "" Then result = defaultValue
    End If
    
    GetSetting = result
End Function

Private Function GetOrCreateSheet(ByVal sheetName As String, Optional ByVal headers As Variant) As Worksheet
    Dim ws As Worksheet, wb As Workbook
    Set wb = ThisWorkbook
    
    On Error Resume Next
    Set ws = wb.Sheets(sheetName)
    On Error GoTo 0
    
    If ws Is Nothing Then
        On Error Resume Next
        Set ws = wb.Sheets.Add(After:=wb.Sheets(wb.Sheets.Count))
        If Err.Number <> 0 Then
            MsgBox "Could not create worksheet '" & sheetName & "'.", vbCritical
            Set GetOrCreateSheet = Nothing
            Exit Function
        End If
        ws.Name = sheetName
        On Error GoTo 0
    End If
    
    If Not IsMissing(headers) Then
        Call EnsureMasterHeader(ws, headers)
    End If
    
    Set GetOrCreateSheet = ws
End Function

Private Sub EnsureMasterHeader(ByVal ws As Worksheet, ByVal headers As Variant)
    ' --- Robust Header Check (Phase 3 Fix) ---
    Dim headerRange As Range
    Dim expectedHeader As String, actualHeader As String
    
    ' Fix for 1-element array from `Array("SourceFile")`
    If Not IsArray(headers) Then
        expectedHeader = headers
    Else
        expectedHeader = headers(LBound(headers))
    End If

    actualHeader = ws.Cells(1, 1).Value
    
    ' Check if header is missing or incorrect
    If actualHeader <> expectedHeader Then
        ws.Rows(1).Insert Shift:=xlDown, CopyOrigin:=xlFormatFromLeftOrAbove
        Set headerRange = ws.Cells(1, 1).Resize(1, UBound(headers) - LBound(headers) + 1)
        headerRange.Value = headers
        headerRange.Font.Bold = True ' Precise bolding
    End If
End Sub
