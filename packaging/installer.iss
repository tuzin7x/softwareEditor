; Instalador Inno Setup 6 (por usuário, sem UAC). Gerar com: iscc packaging\installer.iss
#define AppName "Mobília Studio"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
[Setup]
AppId={{6E1E7C1A-6A0C-4E58-9B7E-4D5F2C0B9A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Mobília Studio
DefaultDirName={localappdata}\Programs\Mobilia Studio
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=MobiliaStudio-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\MobiliaStudio.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; GroupDescription: "Atalhos:"

[Files]
Source: "..\dist\MobiliaStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\MobiliaStudio.exe"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\MobiliaStudio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\MobiliaStudio.exe"; Description: "Abrir o {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Dir: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    Dir := ExpandConstant('{localappdata}\MobiliaStudio');
    if DirExists(Dir) then
      if MsgBox('Remover também o motor de IA, os modelos e os trabalhos salvos (' + Dir + ')?',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(Dir, True, True, True);
  end;
end;
