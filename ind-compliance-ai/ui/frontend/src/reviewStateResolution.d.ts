export function resolveCurrentPage(pageOptions: number[] | undefined, currentPage: number): number

export function resolveSelectedTocSequenceId(input: {
  sequenceIds: string[] | undefined
  activePageSequenceIds: string[] | undefined
  selectedId: string | null
}): string | null

export function resolveRequestedStructuralId(structuralId: string | undefined): string | null
