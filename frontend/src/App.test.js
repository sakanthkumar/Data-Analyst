import { render, screen } from '@testing-library/react';

const mockAxiosInstance = {
  get: jest.fn(),
  post: jest.fn(),
};

jest.mock('axios', () => ({
  create: jest.fn(() => mockAxiosInstance),
  get: jest.fn(),
  post: jest.fn(),
}));

test('renders the dashboard upload prompt', async () => {
  const App = require('./App').default;
  mockAxiosInstance.get.mockResolvedValue({ data: { error: 'No dataset has been uploaded' } });

  render(<App />);
  expect(await screen.findByText(/start new analysis/i)).toBeInTheDocument();
});


