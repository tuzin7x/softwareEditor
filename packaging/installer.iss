; Instalador Inno Setup 6 do 7xAi (por usuário, sem UAC). Opcional: serve para instalar em OUTROS PCs.
; (No PC onde o GERAR_EXE.bat roda, o build_windows.ps1 já instala o 7xAi e cria o atalho sozinho.)
; Gerar com: iscc /DAppVersion=X.Y.Z packaging\installer.iss   ->   dist\7xAi-Setup-<versão>.exe
; Arquivo salvo em UTF-8 com BOM: o Inno Setup só lê os acentos corretamente assim.
;
; O AppId é o mesmo da época do nome "Mobília Studio": o Windows trata como atualização do mesmo
; programa. Os dados (motor de IA de vários GB, modelos e trabalhos) continuam em
; %LOCALAPPDATA%\MobiliaStudio de propósito, para não baixar tudo de novo.
#define AppName "7xAi"
#define ExeName "7xAi.exe"
; A versão NÃO fica escrita aqui: vem de mobilia/__init__.py (fonte única). Passe /DAppVersion=X.Y.Z
; (o build_windows.ps1 e os workflows já fazem isso).
#ifndef AppVersion
  #error Falta a versao: use iscc /DAppVersion=X.Y.Z packaging\installer.iss
#endif
[Setup]
AppId={{6E1E7C1A-6A0C-4E58-9B7E-4D5F2C0B9A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
DefaultDirName={localappdata}\Programs\7xAi
; instalações antigas ficavam em "Programs\Mobilia Studio": usa sempre a pasta nova
UsePreviousAppDir=no
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=7xAi-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#ExeName}
UninstallDisplayName={#AppName}
VersionInfoProductName={#AppName}
VersionInfoDescription=Instalador do {#AppName}
VersionInfoCompany={#AppName}
VersionInfoVersion={#AppVersion}
CloseApplications=yes
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
; marcada por padrão (sem a flag "unchecked"): o ícone 7xAi vai para a Área de Trabalho
Name: "desktopicon"; Description: "Criar o ícone 7xAi na Área de Trabalho"; GroupDescription: "Atalhos:"

[InstallDelete]
; atalhos e pasta do programa da época do nome antigo (os dados em %LOCALAPPDATA%\MobiliaStudio ficam)
Type: files; Name: "{userdesktop}\Mobília Studio.lnk"
Type: files; Name: "{userprograms}\Mobília Studio.lnk"
Type: filesandordirs; Name: "{userprograms}\Mobília Studio"
Type: filesandordirs; Name: "{localappdata}\Programs\Mobilia Studio"

[Files]
Source: "..\dist\7xAi\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#ExeName}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "Abrir o {#AppName}"; Flags: nowait postinstall skipifsilent
; atualização automática (o app roda este instalador com /VERYSILENT ... /AUTOUPDATE e fecha): reabre o 7xAi no fim.
; Sem /AUTOUPDATE (instalação normal ou silenciosa do CI) esta linha não roda.
Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; Flags: nowait; Check: IsAutoUpdate

[Code]
// /AUTOUPDATE: parâmetro passado só pela atualização automática do app (mobilia/ui/updater.py)
function IsAutoUpdate: Boolean;
var I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), '/AUTOUPDATE') = 0 then begin
      Result := True;
      Exit;
    end;
end;

function InitializeSetup: Boolean;
begin
  // na atualização automática, dá tempo para o 7xAi terminar de fechar (ele fecha logo depois de abrir o instalador)
  if IsAutoUpdate then
    Sleep(3000);
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Dir: String;
begin
  if CurUninstallStep = usPostUninstall then begin
    // pasta de dados do 7xAi (nome antigo mantido de propósito: ver o cabeçalho)
    Dir := ExpandConstant('{localappdata}\MobiliaStudio');
    if DirExists(Dir) then
      if MsgBox('Remover também o motor de IA, os modelos e os trabalhos salvos do 7xAi (' + Dir + ')?',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(Dir, True, True, True);
  end;
end;
