import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// jsdom (the simulated browser) has no scrolling; real browsers do
Element.prototype.scrollIntoView = () => {}

// remove the rendered page after each test so tests don't affect each other
afterEach(() => {
  cleanup()
  localStorage.clear()
})
