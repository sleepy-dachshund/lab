# CLAUDE.md - AI Assistant Instructions

## Core Principles
- **Simplicity First**: Every change should be minimal and focused
- **Communicate Clearly**: Provide high-level explanations, not implementation details
- **Document Everything**: Track all decisions and changes in projectplan.md
- **Test Before Committing**: Verify changes work as expected

## Standard Workflow

### 1. Analysis & Planning Phase
- Read and understand the codebase structure
- Identify all relevant files and dependencies
- Create or update `projectplan.md` with:
  - **Problem Statement**: Clear description of what needs to be solved
  - **Approach**: High-level strategy
  - **Todo List**: Specific, checkable tasks (use `- [ ]` format)
  - **Success Criteria**: How we'll know when we're done

### 2. Verification Checkpoint
- Present the plan and wait for approval
- Ask clarifying questions if requirements are ambiguous
- Confirm understanding of edge cases and constraints

### 3. Implementation Phase
- Work through todo items systematically
- Mark items complete with `- [x]` as you go
- For each change:
  - Provide a brief summary (1-2 sentences)
  - Only show code if specifically requested
  - Focus on WHAT changed, not HOW
- Follow these constraints:
  - Prefer small, isolated changes
  - Avoid refactoring unrelated code
  - Maintain existing code style
  - Add comments only for complex logic
- Cose Style
  - Use type hinting in function definitions (for params and output) 
  - Use numpy docstrings for function documentation
  - Include description of dataframe outputs in docstring if applicable (index and column descriptions)
  - Prefer small, modular functions over large monolithic ones

### 4. Testing & Validation
- Run relevant tests if available
- Manually verify critical paths
- Check for obvious edge cases
- Report any test failures immediately

### 5. Review & Documentation
- Add a `## Review` section to projectplan.md including:
  - Summary of completed changes
  - Any deviations from original plan
  - Known limitations or future improvements
  - Files modified (list format)

## Communication Guidelines
- Default to high-level summaries
- Ask for permission before large changes
- Flag any security or performance concerns immediately
- Use this format for updates:
  ```
  ✅ [Task]: [What was accomplished]
  📝 Changed: [Files affected]
  ⚠️ Note: [Any important considerations]
  ```

## When to Stop and Ask
- Architectural decisions that affect multiple systems
- Changes requiring new dependencies
- Modifications to authentication/authorization
- Database schema changes
- Any destructive operations
- When the simple approach seems inadequate

## File Organization
- Keep related changes in the same commit
- Create new files in appropriate directories
- Follow existing naming conventions
- Update imports/exports as needed

## Error Handling
- Don't suppress errors silently
- Add basic error handling for user-facing features
- Log errors appropriately
- Fail fast for development issues

## Tools & Commands
- Use filesystem operations for reading/writing
- Prefer built-in language features over external libraries
- Document any new dependencies in projectplan.md
- Include setup instructions if configuration changes