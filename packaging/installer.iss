; Inno Setup — Reaction AutoEdit (per-user install, no admin needed)
#ifndef Edition
  #define Edition "cpu"
#endif
[Setup]
AppName=Reaction AutoEdit
AppVersion={#GetEnv("RAE_VERSION")}
DefaultDirName={localappdata}\Programs\ReactionAutoEdit
PrivilegesRequired=lowest
OutputBaseFilename=ReactionAutoEdit-{#GetEnv("RAE_VERSION")}-{#Edition}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\ReactionAutoEdit\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{userprograms}\Reaction AutoEdit"; Filename: "{app}\ReactionAutoEdit.exe"
Name: "{userdesktop}\Reaction AutoEdit"; Filename: "{app}\ReactionAutoEdit.exe"

[Run]
Filename: "{app}\ReactionAutoEdit.exe"; Description: "Launch Reaction AutoEdit"; Flags: postinstall nowait skipifsilent
