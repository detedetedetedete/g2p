# -*- mode: python -*-

block_cipher = None


a = Analysis(['g2p-master.py'],
             pathex=['/home/deiwid/VGTU/mag/g2p-rework'],
             binaries=[],
             datas=[],
             hiddenimports=['SimpleWebSocketServer'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='g2p-master',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
