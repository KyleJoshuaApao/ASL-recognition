$ErrorActionPreference = 'Stop'

Write-Host "Downloading ODBC Driver 17..."
Invoke-WebRequest -Uri "https://go.microsoft.com/fwlink/?linkid=2202804" -OutFile "msodbcsql.msi"

Write-Host "Installing ODBC Driver 17..."
Start-Process "msiexec.exe" -ArgumentList "/i msodbcsql.msi /qn IACCEPTMSODBCSQLLICENSETERMS=YES" -Wait -NoNewWindow

Write-Host "Downloading SQL Server 2019 Express..."
Invoke-WebRequest -Uri "https://download.microsoft.com/download/7/c/1/7c14e92e-bdcb-4f89-b7cf-93543e7112d1/SQLEXPR_x64_ENU.exe" -OutFile "SQLEXPR.exe"

Write-Host "Extracting SQL Server..."
Start-Process ".\SQLEXPR.exe" -ArgumentList "/q /x:$pwd\SQLEXPR_EXTRACT" -Wait -NoNewWindow

Write-Host "Installing SQL Server Express (This will take a few minutes)..."
Start-Process "$pwd\SQLEXPR_EXTRACT\SETUP.EXE" -ArgumentList "/Q /IACCEPTSQLSERVERLICENSETERMS /ACTION=install /FEATURES=SQL /INSTANCENAME=SQLEXPRESS /SQLSVCACCOUNT=`"NT AUTHORITY\Network Service`" /SQLSYSADMINACCOUNTS=`"BUILTIN\ADMINISTRATORS`"" -Wait -NoNewWindow

Write-Host "Installation completed."
