from pathlib import Path

def check_empty_dirs_with_symlinks(base_path: Path):
    empty_dirs = []

    for d in base_path.iterdir():
        if d.is_dir():
            # ディレクトリの中身をリストアップ（リンク含む）
            contents = list(d.iterdir())
            
            if not contents:
                # 中身が完全にない場合
                empty_dirs.append(d)
            else:
                # サブディレクトリやリンクがある場合も再帰チェック
                all_empty = True
                for f in contents:
                    if f.is_dir():
                        sub_empty = check_empty_dirs_with_symlinks(f)
                        if sub_empty:
                            empty_dirs.extend(sub_empty)
                        else:
                            all_empty = False
                    elif f.is_file() or f.is_symlink():
                        # ファイルやリンクがあれば空ではない
                        all_empty = False
                if all_empty:
                    empty_dirs.append(d)
    return empty_dirs

base_dir = Path("/workspaces/template_pytorch/isee_dataset")
empty_dirs = check_empty_dirs_with_symlinks(base_dir)

if empty_dirs:
    print("空のディレクトリ（サブディレクトリ＆シンボリックリンク含む）:")
    for d in empty_dirs:
        print(d)
else:
    print("空のディレクトリはありません。")
