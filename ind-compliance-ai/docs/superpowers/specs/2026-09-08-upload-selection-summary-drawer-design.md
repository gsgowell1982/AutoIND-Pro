# Upload Selection Summary and File Detail Drawer

## Goal

Keep the materials upload panel stable when an eCTD folder contains many files while preserving an auditable way to inspect every selected path before submission.

## Product decision

Project-mode uploaders (`zip` and `directory`) will not render the upload library's raw file list inline. The left upload column will show a compact selection summary and a `View file list` action. The action opens a bounded drawer containing searchable relative paths, file sizes, and the selected-file count. The right project tree remains the canonical hierarchy view and continues to show directories, empty directories, and files during queued/processing states.

Single-document mode keeps the existing compact uploader file item because there is no project tree duplication and the selected document is the primary review object.

## Boundaries

- This is a frontend presentation change only; upload multipart fields and job APIs do not change.
- The summary derives counts from the browser selection and explicit directory manifest already held by the component.
- The drawer is read-only for project selections. Re-selecting the package remains the safe way to change package membership.
- Successful processing and review workbench behavior remain unchanged.

## Interaction and layout

- Project mode: `Upload.Dragger` uses `showUploadList={false}`.
- The summary exposes mode, file count, directory count, root label when available, and a searchable detail action.
- The drawer has its own scroll surface and does not change the upload card height.
- Search is case-insensitive and matches the normalized relative path or filename.
- Empty selections show a clear empty state; no fake file rows are created for explicit empty directories.

## Verification

- Pure helper tests cover project counts, directory de-duplication, root labeling, byte totals, and case-insensitive path filtering.
- TypeScript production builds must pass in the primary and mirror repositories.
- Existing eCTD intake/inventory/support/rule/upload tests and workbench tests must remain green.
